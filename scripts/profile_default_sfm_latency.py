#!/usr/bin/env python3
"""Offline latency diagnosis; live server access is limited to GET /health."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summary(samples):
    ordered = sorted(samples)
    return dict(n=len(samples), median_ms=statistics.median(samples),
                mean_ms=statistics.mean(samples), min_ms=min(samples),
                max_ms=max(samples), p90_ms=ordered[min(len(ordered)-1, int(.9*len(ordered)))])


def transport(args):
    import numpy as np
    import requests
    from server_flower_rpc import verify_bundle
    from deployment.default_sfm.reference.Real_Robo import env_rpc_server as wire

    report = {}
    for persistent in (True, False):
        timings = []
        with requests.Session() as session:
            session.trust_env = False
            for _ in range(args.samples + 1):
                started = time.perf_counter()
                response = session.get(args.health_url, timeout=5,
                                       headers={} if persistent else {'Connection': 'close'})
                response.raise_for_status()
                response.json()
                timings.append(1000*(time.perf_counter()-started))
        report['live_health_keepalive' if persistent else 'live_health_close'] = summary(timings[1:])
    bundle, manifest = verify_bundle(ROOT/'datasets/default_sfm_crop_v2/rpc_bundle')
    observation = {key: np.zeros(shape, dtype=np.uint8 if key.startswith('rgb_') else np.float32)
                   for key, shape in bundle.stats['schema']['shapes'].items()}
    body = wire.pack_arrays(observation)
    report['query_payload_bytes'] = len(body)

    class Dummy:
        simulation_only = True

        def predict(self, observation):
            time.sleep(.060)
            return np.zeros((10, 7), dtype=np.float32)

    for nodelay in (False, True):
        handler = wire.create_handler(bundle, Dummy())
        handler.disable_nagle_algorithm = nodelay
        with ThreadingHTTPServer(('127.0.0.1', 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            timings = []
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    session.headers.update({'X-Policy-Protocol': wire.PROTOCOL,
                                            'X-Preprocess-Id': manifest['bundle_id'],
                                            'Content-Type': wire.MEDIA_TYPE})
                    for i in range(args.samples + 1):
                        started = time.perf_counter()
                        response = session.post(f'http://127.0.0.1:{server.server_port}/query',
                                                data=body, headers={'X-Request-Id': str(i)}, timeout=5)
                        response.raise_for_status()
                        wire.unpack_arrays(response.content)
                        timings.append(1000*(time.perf_counter()-started))
            finally:
                server.shutdown()
                thread.join(timeout=2)
        report[f'offline_dummy_60ms_nodelay_{nodelay}'] = summary(timings[1:])
    return report


def model_profile(args):
    import numpy as np
    import torch
    import server_flower_rpc as rpc

    bundle_path = ROOT/'datasets/default_sfm_crop_v2/rpc_bundle'
    bundle, manifest = rpc.verify_bundle(bundle_path)
    load_args = SimpleNamespace(checkpoint=rpc.DEFAULT_CHECKPOINT, training_config=None,
                                device='cuda', precision='float16', cpu_threads=4,
                                vlm_path=None, weights='ema', preprocess_bundle=bundle_path,
                                seed=42, task=None)
    policy, audit = rpc.load_policy(load_args, bundle, manifest)
    policy.reset('usb')
    rng = np.random.default_rng(42)
    observation = {key: (rng.integers(0, 256, shape, dtype=np.uint8) if key.startswith('rgb_')
                         else np.zeros(shape, dtype=np.float32))
                   for key, shape in bundle.stats['schema']['shapes'].items()}
    report = dict(audit=audit, gpu=torch.cuda.get_device_name(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch_version=torch.__version__, sampling_steps=policy.model.num_sampling_steps,
                  input='synthetic RGB 224x224 x2; zero normalized state and force',
                  stages={}, warmup=5)

    def measure(label, count):
        timings = []
        for _ in range(count):
            torch.cuda.synchronize()
            started = time.perf_counter()
            policy.predict(observation)
            torch.cuda.synchronize()
            timings.append(1000*(time.perf_counter()-started))
        report[label] = summary(timings)
        print(label, json.dumps(report[label]), flush=True)

    for _ in range(5):
        policy.predict(observation)
    measure('original_full_predict', args.samples)
    if args.ablations:
        state = policy.generator.get_state()
        expected = policy.predict(observation)
        policy.model.requires_grad_(True)
        policy.generator.set_state(state)
        actual = policy.predict(observation)
        report['cache_eligible_max_abs_difference'] = float(np.max(np.abs(actual-expected)))
        for _ in range(3):
            policy.predict(observation)
        measure('cache_eligible_parameters_inference_mode', args.samples)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            for _ in range(3):
                policy.predict(observation)
            measure('cache_eligible_persistent_autocast', args.samples)
        policy.model.requires_grad_(False)
        measure('original_recheck', args.samples)
        report['ablation_scope'] = 'Offline model only; parameter values unchanged; inference_mode disables gradients'
        return report
    records = []

    @contextmanager
    def stage(label):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        yield
        end.record()
        records.append((label, begin, end, 1000*(time.perf_counter()-started)))

    originals = []

    def wrap(obj, name, label):
        original = getattr(obj, name)
        originals.append((obj, name, original))

        def timed(*values, **kwargs):
            with stage(label):
                return original(*values, **kwargs)
        setattr(obj, name, timed)

    wrap(policy.model, 'encode_observations', 'encode_total')
    wrap(policy.model, 'encode_force_history', 'force_history')
    wrap(policy.model.vlm, '_encode_image', 'vision_each_view')
    wrap(policy.model, '_get_text_embeddings', 'text_embeddings')
    wrap(policy.model.vlm.get_encoder(), 'forward', 'vlm_encoder')
    wrap(policy.model, 'sample_actions', 'sample_total')
    wrap(policy.model, 'dit_forward', 'dit_each_step')
    wrap(rpc, 'observation_batch', 'cpu_preprocess')
    for _ in range(args.samples):
        policy.predict(observation)
    torch.cuda.synchronize()
    stage_samples = defaultdict(list)
    cpu_samples = defaultdict(list)
    for label, begin, end, cpu_ms in records:
        stage_samples[label].append(begin.elapsed_time(end))
        cpu_samples[label].append(cpu_ms)
    for label, values in stage_samples.items():
        report['stages'][label] = dict(cuda_stream_ms=summary(values),
                                       host_call_ms=summary(cpu_samples[label]))
    for obj, name, original in reversed(originals):
        setattr(obj, name, original)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as profiler:
        for _ in range(3):
            policy.predict(observation)
            profiler.step()
    table = profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=25)
    report['profiler_cuda_table'] = table
    report['profiler_cpu_table'] = profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=25)
    report['cuda_sync_events'] = [dict(key=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total,
                                       cpu_us=e.cpu_time_total)
                                  for e in profiler.key_averages()
                                  if any(term in e.key.lower() for term in ['synchroniz', 'memcpy', 'item', 'scalar'])]
    report['peak_gpu_memory_mib'] = torch.cuda.max_memory_allocated()/2**20
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['transport', 'model'])
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--ablations', action='store_true', help='Offline autocast-cache comparison instead of profiling')
    parser.add_argument('--health-url', default='http://127.0.0.1:45587/health')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error('--samples must be at least 2')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), mode=args.mode,
                  live_policy_modified=False)
    report.update(transport(args) if args.mode == 'transport' else model_profile(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
