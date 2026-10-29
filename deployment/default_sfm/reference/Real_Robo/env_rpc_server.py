"""Standalone reference server for the offline HTTP/NPZ policy contract.

Run this file directly (it can be copied outside the Real_Robo package)::

    python env_rpc_server.py --preprocess-bundle /path/to/bundle --port 45587

Dependencies: Python standard library and NumPy; h5py is only needed when
--preprocess-bundle points to an HDF5 file instead of a bundle directory.
The preprocessing config and normalization statistics must still be supplied.
No robot, preprocessing, model, or local package imports are required.

Endpoints:
    GET /health  - JSON metadata, including the bundle ID and action shape.
    POST /reset  - JSON episode context for an optional policy.reset method.
    POST /query  - Preprocessed observation arrays in uncompressed NPZ format;
                   returns one normalized `actions` array of shape [H, A].
                   H may differ from the stored bundle chunk length;
                   /health continues to describe the source data bundle.

POST requests require X-Policy-Protocol and X-Preprocess-Id from /health.
Queries also require an ASCII alphanumeric X-Request-Id (1..128 characters).

To adapt this reference for a trained model, replace DummyPolicy in main()
with an object implementing predict(observation), returning normalized [H, A]
actions. An optional reset(text=..., previous_episode=...) handles new episodes.
The included dummy returns normalized zeros, NOT physical hold commands, and
advertises simulation_only=True so the robot client rejects execution.

The wire contract is kept compatible with Real_Robo/inference/protocol.py;
all server-side implementation is intentionally included below for reference.
"""
import argparse
import hashlib
import json
import logging
import math
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import numpy as np


# Bundle configuration and normalization metadata (no preprocessing).


class InferenceBundle:
    def __init__(self, path):
        path = Path(path).expanduser()
        if path.suffix in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(path) as f:
                self.config = json.loads(f['metadata/inference_config'][()])
                self.stats = json.loads(f['metadata/normalization'][()])
        else:
            self.config = json.loads((path/'inference_config.json').read_text())
            self.stats = json.loads((path/self.config['normalization_file']).read_text())
        action = dict(self.config['actions'])
        action.pop('declared_formats', None)
        schema = self.stats.get('schema', {})
        if (self.config.get('version') != 1 or self.stats.get('version') != 1 or
                schema.get('observations') != self.config['observations'] or
                schema.get('action') != action):
            raise ValueError('Inference config and normalization statistics do not match')
        shape = schema.get('action_shape', [])
        if (len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape) or
                shape != [action['length'], 7 * len(action['sides'])]):
            raise ValueError('Invalid bundle action shape')
        if set(schema.get('shapes', {})) != {e['target'] for e in self.config['observations']}:
            raise ValueError('Observation shapes differ from the bundle keys')


# HTTP/NPZ wire contract, validation, and safe numeric array encoding.


PROTOCOL = 'clean-share-policy-v2'

TRANSPORT = 'http-npz'

MEDIA_TYPE = 'application/x-npz'

MAX_PAYLOAD_BYTES = 64 * 1024**2

def bundle_id(preprocessor):
    data = dict(config=preprocessor.config, normalization=preprocessor.stats)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def metadata(preprocessor):
    """Describe the source data bundle; action_shape is its stored chunk shape."""
    contract = preprocessor.config['actions']
    return dict(protocol=PROTOCOL, transport=TRANSPORT, bundle_id=bundle_id(preprocessor),
                observations='preprocessed', actions='normalized',
                action_shape=preprocessor.stats['schema']['action_shape'],
                action_contract=contract, simulation_only=False)

def validate_observation(preprocessor, observation):
    shapes = preprocessor.stats['schema']['shapes']
    if set(observation) != set(shapes):
        raise ValueError(f'Observation keys must be {sorted(shapes)}, got {sorted(observation)}')
    result = {}
    images = {e['target'] for e in preprocessor.config['observations'] if e['type'] == 'image'}
    for key, shape in shapes.items():
        value = np.asarray(observation[key])
        if list(value.shape) != shape or value.dtype.kind not in 'fiu' or not np.all(np.isfinite(value)):
            raise ValueError(f'Invalid observation {key}: expected finite {shape}, got {value.shape}')
        if key in images:
            if np.any(value < 0) or np.any(value > 255) or np.any(value != np.floor(value)):
                raise ValueError(f'{key} must contain RGB uint8 pixel values')
            value = value.astype(np.uint8)
        else:
            value = value.astype(np.float32)
        result[key] = value
    return result

def validate_actions(preprocessor, actions):
    """Validate a live response [H, A], independent of the dataset horizon."""
    value = np.asarray(actions, dtype=np.float32)
    action_dim = preprocessor.stats['schema']['action_shape'][1]
    if (value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != action_dim or
            not np.all(np.isfinite(value))):
        raise ValueError(f'Expected finite normalized actions [H, {action_dim}] with H >= 1, got {value.shape}')
    if value.nbytes > MAX_PAYLOAD_BYTES:
        raise ValueError('RPC payload exceeds 64 MiB')
    return value

def pack_arrays(arrays):
    """One uncompressed NPZ format for observations AND action chunks."""
    if not isinstance(arrays, dict) or not arrays or len(arrays) > 128:
        raise ValueError('RPC payload must be a nonempty dict of up to 128 numeric arrays')
    buffer = BytesIO()
    # Writing members explicitly avoids np.savez reserved keyword collisions
    # and permits allow_pickle=False on NumPy 1.x as well as NumPy 2.x.
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
        for key, values in arrays.items():
            if not isinstance(key, str) or not key or '\x00' in key:
                raise ValueError('Array keys must be nonempty strings without NUL')
            value = np.asarray(values)
            if value.dtype.kind not in 'biuf':
                raise ValueError(f'RPC array {key} must have a numeric dtype, got {value.dtype}')
            if value.nbytes + buffer.tell() > MAX_PAYLOAD_BYTES:
                raise ValueError('RPC payload exceeds 64 MiB')
            with archive.open(key + '.npy', 'w') as member:
                np.lib.format.write_array(member, value, allow_pickle=False)
    data = buffer.getvalue()
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError('RPC payload exceeds 64 MiB')
    return data

def unpack_arrays(data):
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError('RPC payload exceeds 64 MiB')
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            members = archive.infolist()
            if not 1 <= len(members) <= 128 or sum(m.file_size for m in members) > MAX_PAYLOAD_BYTES:
                raise ValueError('Invalid NPZ member count or expanded payload size')
            result = {}
            for member in members:
                key = member.filename[:-4]
                if (not member.filename.endswith('.npy') or not key or key in result or
                        member.compress_type != zipfile.ZIP_STORED):
                    raise ValueError('Expected unique, uncompressed NPY members')
                contents = archive.read(member)
                stream = BytesIO(contents)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported NPY version')
                # Validate the declared allocation before numpy loads the array.
                if (dtype.kind not in 'biuf' or
                        math.prod(shape) * dtype.itemsize != len(contents) - stream.tell()):
                    raise ValueError('Invalid numeric NPY shape, dtype or data length')
                stream.seek(0)
                result[key] = np.load(stream, allow_pickle=False)
            return result
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        raise ValueError('Invalid NPZ payload') from exc


# Replace this policy with a model adapter when using this as a template.


class DummyPolicy:
    simulation_only = True

    def __init__(self, bundle):
        self.shape = tuple(bundle.stats['schema']['action_shape'])

    def predict(self, observation):
        # These are normalized zeros, not physical hold commands.
        return np.zeros(self.shape, dtype=np.float32)


# HTTP endpoints and serialized policy calls.


class RequestError(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)

def create_handler(bundle, policy):
    contract = metadata(bundle)
    contract['simulation_only'] = bool(getattr(policy, 'simulation_only', False))
    policy_lock = Lock()

    class PolicyHandler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.settimeout(10)  # Idle / body I/O; not a model execution deadline.

        def log_message(self, format, *args):
            pass  # No per-observation console logging in the inference path.

        def respond(self, status, body, content_type='application/json', headers=None, close=False):
            if close:
                self.close_connection = True
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Policy-Protocol', PROTOCOL)
            self.send_header('Cache-Control', 'no-store')
            if close:
                self.send_header('Connection', 'close')
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def json_response(self, status, value, close=False):
            self.respond(status, json.dumps(value, allow_nan=False).encode(), close=close)

        def send_error(self, code, message=None, explain=None):
            # Close on rejection: unread request bytes must never become a next request.
            self.json_response(code, {'ok': False, 'error': message or self.responses.get(code, ('HTTP error',))[0]}, close=True)

        def handle_expect_100(self):
            self.send_error(417, 'Expect: 100-continue is not supported')
            return False

        def read_body(self):
            if self.headers.get_all('Transfer-Encoding'):
                raise RequestError(400, 'Transfer-Encoding is not supported; send Content-Length')
            lengths = self.headers.get_all('Content-Length', [])
            if not lengths:
                raise RequestError(411, 'Content-Length is required')
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                raise RequestError(400, 'Invalid Content-Length')
            size = int(lengths[0])
            if size > MAX_PAYLOAD_BYTES:
                raise RequestError(413, 'RPC payload exceeds 64 MiB')
            try:
                body = self.rfile.read(size)
            except TimeoutError as exc:
                raise RequestError(408, 'Request body timed out') from exc
            if len(body) != size:
                raise RequestError(400, 'Incomplete request body')
            return body

        def do_GET(self):
            try:
                if urlsplit(self.path).path != '/health':
                    raise RequestError(404, 'Unknown endpoint')
                if self.headers.get_all('Transfer-Encoding') or self.headers.get_all('Content-Length', []) not in ([], ['0']):
                    raise RequestError(400, 'GET /health must not contain a body')
                self.json_response(200, dict(ok=True, **contract))
            except (ValueError, RequestError) as exc:
                self.send_error(getattr(exc, 'status', 400), str(exc))
            except (ConnectionError, TimeoutError):
                self.close_connection = True

        def do_POST(self):
            try:
                path = urlsplit(self.path).path
                if path not in ('/query', '/reset'):
                    raise RequestError(404, 'Unknown endpoint')
                if self.headers.get('X-Policy-Protocol') != PROTOCOL:
                    raise RequestError(409, 'Incompatible X-Policy-Protocol; update both endpoints')
                if self.headers.get('X-Preprocess-Id') != contract['bundle_id']:
                    raise RequestError(409, 'Missing or incompatible X-Preprocess-Id')
                content_type = self.headers.get_content_type()
                if path == '/query':
                    if content_type != MEDIA_TYPE:
                        raise RequestError(415, f'Expected {MEDIA_TYPE}')
                    request_id = self.headers.get('X-Request-Id', '')
                    if not 1 <= len(request_id) <= 128 or not request_id.isascii() or not request_id.isalnum():
                        raise RequestError(400, 'X-Request-Id must contain 1..128 ASCII letters/digits')
                    observation = validate_observation(bundle, unpack_arrays(self.read_body()))
                    with policy_lock:
                        actions = validate_actions(bundle, policy.predict(observation))
                        result = pack_arrays({'actions': actions})
                    self.respond(200, result, MEDIA_TYPE, {'X-Request-Id': request_id})
                else:
                    if content_type != 'application/json':
                        raise RequestError(415, 'Expected application/json')
                    body = json.loads(self.read_body())
                    if not isinstance(body, dict):
                        raise RequestError(400, 'Reset payload must be a JSON object')
                    with policy_lock:
                        if hasattr(policy, 'reset'):
                            policy.reset(text=body.get('text', ''), previous_episode=body.get('previous_episode'))
                    self.json_response(200, {'ok': True})
            except RequestError as exc:
                self.send_error(exc.status, str(exc))
            except (ValueError, KeyError, TypeError) as exc:
                self.send_error(400, str(exc))
            except (ConnectionError, TimeoutError):
                self.close_connection = True
            except Exception:
                logging.exception('Policy request failed')
                self.send_error(500, 'Policy inference failed; see server log')

    return PolicyHandler

def serve_policy(bundle, policy, *, host='127.0.0.1', port=45587):
    handler = create_handler(bundle, policy)
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f'Policy bundle: {metadata(bundle)["bundle_id"]}', flush=True)
        print(f'HTTP/NPZ server: http://{host}:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preprocess-bundle', required=True, type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=45587)
    args = parser.parse_args()
    bundle = InferenceBundle(args.preprocess_bundle)
    serve_policy(bundle, DummyPolicy(bundle), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
