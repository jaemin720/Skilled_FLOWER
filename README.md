# FLOWER Calvin Custom

This repository is a modified version of the FLOWER VLA codebase, adapted for custom robot dataset training and inference.

## Overview

This project is based on the original FLOWER VLA repository and includes modifications for custom data preprocessing, datamodule configuration, training scripts, and inference server utilities.

## Main Modifications

- Added custom dataset preprocessing scripts
- Added custom Calvin datamodule configurations
- Modified FLOWER model and training pipeline for custom robot data
- Added training scripts for custom dataset experiments
- Added inference server scripts for evaluation

## Custom Training Without SkillVAE

`train_custom_alltrain.sh` uses `config_custom_calvin_alltrain` to train the basic FLOWER policy with image, language, and proprioception inputs:

```bash
bash train_custom_alltrain.sh
```

SkillVAE loading, action-history conditioning (including its MLP/Conv1d variants and proprioception fusion), and alternating proprioception/skill conditioning have been removed from both model implementations. The dataset no longer reads the previous 32 actions or gripper states, or adds `skill_prev_*` fields to batches. The SkillVAE module and training configuration have also been removed.

The existing force-history and auxiliary BBox options remain available. SkillVAE-trained checkpoints contain additional parameters and cannot be resumed with strict checkpoint loading in this version; start a new run from the base FLOWER pretrained weights.

The base checkpoint has been downloaded from [mbreuss/flower_vla_pret](https://huggingface.co/mbreuss/flower_vla_pret/tree/ffd4134a2167a2be32bd7ad93455ed78dc2dcda2) to `flower_vla_pret/360000_model_weights.pt` (1.67 GB). Both `conf/model/flower.yaml` and `conf/config_custom_calvin_alltrain.yaml` point to `/home/jack/Skilled_FLOWER/flower_vla_pret/360000_model_weights.pt`.

The file's SHA-256 matches the upstream value: `f654fe4adc6a348e9c06424a3444fddf700820818b0b81b917db0ccec94eff60`. The download revision and checksum are recorded in `flower_vla_pret/source.json`; the upstream config and model card are saved alongside the weights. A shape check against the current model matched all 295 core policy tensors. Custom proprioception layers absent from the checkpoint or with different input dimensions retain their new initialization. This validation did not run full Florence inference or GPU training.

## Default SFM real-robot dataset

`preprocess/default_sfm/convert.py` adapts the pinned TurboVLA crop-v2 recipe to FLOWER's disk dataset. It selects only the nine tasks' top-level `task_p1` through `task_p5` folders, requires ten demonstrations per position (50 per task, 450 total), and excludes `bnc_old`, `bnc_test`, `test`, and `usb_insert`. The current BNC recordings are used; the historical TurboVLA copy manifest and frame counts are not reused.

Run the converter on Kingston, where the 524 GB of original pickles are stored. The output must be on the large data disk, outside the raw dataset:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/csilab/miniforge3/envs/force_flow/bin/python \
  /home/csilab/share_projects/clean_share/preprocess_runs/flower_20261028/default_sfm/convert.py \
  --raw-root /home/csilab/share_projects/clean_share/demos/default_sfm \
  --output /media/csilab/robot_data/preprocessed/flower_default_sfm_crop_v2_20261028 \
  --workers 3 --resume
```

The converter verifies initial-motion trimming, held gripper targets, and every cropped image against the archived reference implementations. It reconstructs gripper commands before trimming, preserves later pauses and the final segment, and fits fresh statistics over all retained training data. It stores 224×224 RGB views, standardized 13-D proprioception, standardized 50×6 F/T histories, and normalized 7-D delta actions. Translation is in the robot base frame; rotation is the target-versus-measured rotation vector. Roll and yaw state values are wrapped before normalization. State/F/T are standardized; the six pose action components use `symlog_minmax` (temperature 20); gripper targets stay at −1/+1. These are **not** raw absolute robot commands: inference must invert `normalization.json` and reconstruct absolute targets as specified in `data_contract.json`.

The generated `archives/` contains one lossless NPZ tar archive per task, metadata, and SHA-256 checksums. Transfer only these archives to this server, then unpack and verify:

```bash
python scripts/unpack_default_sfm.py \
  --archives datasets/default_sfm_crop_v2_archives \
  --output datasets/default_sfm_crop_v2
/home/jack/miniforge3/envs/flower/bin/python scripts/verify_default_sfm.py
```

The local dataset is `datasets/default_sfm_crop_v2`. `flower_verification.json` records the actual loader checks. Source hashes, per-demo provenance, normalization statistics, and task instructions are retained alongside the dataset. Language embeddings are unused compatibility placeholders; FLOWER encodes the exact task text using Florence.

To start training explicitly:

```bash
bash train_default_sfm.sh
```

`train_default_sfm.sh` defaults to GPUs 0 and 1 (`CUDA_VISIBLE_DEVICES=0,1`) with two DDP workers; an explicit environment variable can override the GPU selection. `conf/config_default_sfm.yaml` uses all 450 demonstrations for training, a configurable action horizon (default 10), both camera views, all 13 state components, and GRU-based F/T conditioning in the base FLOWER DiT. There is no validation split. `SfmDiskDataset` samples all complete windows within each demonstration, including the final valid window, and uses extracted actions to read an entire action chunk without loading its future image files. It applies shared color jitter to both views while preserving the letterbox padding. The dedicated transforms apply no additional crop and no second numeric normalization. The existing custom CALVIN configs remain separate.

Set `act_seq_len` in `conf/config_default_sfm.yaml` or override it on the command line. This controls both the loader's action chunk and the model's `act_window_size`; every per-frame action is already stored, so changing the horizon requires no new preprocessing. All 450 demos have at least 193 frames. With 10 actions per chunk, the loader has 133,189 complete training windows and returns actions shaped `[B,10,7]`. Windows never cross demonstration boundaries; an episode shorter than the requested window raises an error.

```bash
bash train_default_sfm.sh act_seq_len=10
# A different training horizon uses the same data and normalization.
bash train_default_sfm.sh act_seq_len=50
```

At the 10 Hz control rate, 10 future actions cover about 1 second. `multistep` separately controls how many predicted actions are executed before replanning during rollout; its default remains 5 and it must not exceed the predicted chunk length. The 50-sample F/T history is independent of `act_seq_len` and remains about 100 ms at 500 Hz.

SFM checkpoints are saved after every five completed epochs (5, 10, 15, ...), retaining all periodic checkpoints. Checkpoints include optimizer and EMA state. `save_last` is disabled because Lightning 2.0 would otherwise write `last.ckpt` at every epoch regardless of the five-epoch interval. The epoch value in filenames uses Lightning's zero-based numbering. `batch_size` is per GPU: 32/64 per GPU corresponds to a global batch of 64/128 on two GPUs with no gradient accumulation.

`scripts/benchmark_default_sfm.py` measures full-model DDP memory on real SFM batches without optimizer updates or checkpoint saves. It builds the actual Florence architecture from cached config/code/tokenizer files and loads all 604 VLM tensors from `flower_vla_pret/360000_model_weights.pt`; it does not load separate Florence backbone weights. AdamW moments, EMA weights, and conservative update workspace are included. On GPUs 0/1 (RTX PRO 6000 Blackwell Max-Q), with both 224x224 views, FP16 mixed precision, and `act_seq_len=10`, the measured per-GPU peaks were:

| Per-GPU batch | Global batch | Peak allocated VRAM | Peak reserved VRAM | Forward/backward |
| --- | --- | --- | --- | --- |
| 32 | 64 | 53.8 GiB | 54.7 GiB | Passed; finite gradients |
| 64 | 128 | 87.0 GiB | 88.3 GiB | Passed; finite gradients |

The report is `.benchmarks/default_sfm_batch_memory.json`. Batch 64 fits the approximately 95 GiB accessible memory with about 6.7 GiB remaining after allocator reservations. These are short memory diagnostics, not a completed training run or a measurement of data-loading/optimizer throughput. The active SFM configuration selects **64 per GPU (global batch 128)** for **20 epochs**, with 1,041 steps per epoch and a 20,820-step LR schedule. The schedule warms up for 1,041 steps from 2e-6 to 2e-5, holds for 2,082 steps, then decays for 17,697 steps to 1e-5. Recompute the schedule length if the batch size, dataset, or epoch count changes.

`scripts/benchmark_default_sfm_time.py` measures loader, transfer, forward/backward, AdamW, and EMA time while applying AdamW only to separate diagnostic copies. With five warmup and 30 measured steps, per-GPU batches 32/64 took 0.592/1.026 seconds per step, estimating 6h51m/5h56m for 20 epochs before checkpoint and external logging overhead. The measurements are saved in `.benchmarks/default_sfm_batch_time.json`.

SFM training uses the pinned local Florence snapshot and the existing FLOWER checkpoint. WandB runs offline. `TrainingStatus` writes `training_status.json` and `training_progress.jsonl` under the trainer's run directory every 20 steps, recording loss, LR, AMP scale, GPU memory, and evidence of GRU parameter updates. The current detached run's location is recorded in `logs/default_sfm/latest_run.json`.

Older dataset contracts contain `action_horizon: 5` as a record of the conversion recipe. This is not a storage limit. The action normalization uses per-axis minima/maxima, which are unchanged by the conversion recipe's repeated valid chunk entries; changing the training horizon does not require refitting these statistics. New conversions record that recipe length as `normalization_fit_action_horizon` and identify the training config as the source of the runtime horizon.

The F/T input is `right_force_history`: 50 six-channel samples at 500 Hz (about 100 ms), stored oldest to newest and already standardized. The latest observation's complete history passes through a unidirectional, one-layer GRU (hidden size 128), then a linear projection and LayerNorm to the DiT width (1024). The GRU starts from zero for every window. Its embedding is added to the timestep, 10 Hz control-frequency, and proprioception embeddings, conditioning both the shared AdaLN controller and every DiT block's AdaLN scale, shift, and gates. The 500 Hz sensor rate does not change the 10 Hz action rate. The GRU runs once per observation and its embedding is reused across flow sampling steps, with gradients retained during training.

The SFM loader rejects additional crop, resize, or spatial-shift transforms for the already-cropped images. Loader transforms are limited to pixel scaling and normalization; the shared color jitter preserves image geometry and restores the original black padding. Florence receives these 224x224 tensors directly, bypassing its image processor.

The SFM config requires `[B,50,6]` or `[B,obs_seq_len,50,6]` F/T inputs (`[50,6]` also works for single-observation inference). Missing histories and incorrect lengths raise errors; zero-valued histories remain valid. Start this GRU model from the base FLOWER pretrained weights. Legacy Conv1d checkpoints remain loadable using their saved hyperparameters; for an explicit legacy configuration, set `model.force_history_encoder_type=conv1d` and restore its original hidden size, convolution, and dropout settings.

The stored state order is `gripper,x,y,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz`. These 13 proprioception channels already exclude all force/torque values; F/T is supplied only through the GRU history branch. Optional state ablations need only configuration changes:

```bash
# Dedicated X/Y-only ablation, with separate logs under logs/default_sfm_no_xy.
bash train_default_sfm.sh --config-name config_default_sfm_no_xy
# Remove x/y, preserving gripper and all velocities (11 dimensions).
bash train_default_sfm.sh proprio_dims=11 \
  'datamodule.proprioception_dims.keep_indices=[[0,1],[3,13]]'
# Remove x/y and vx/vy (9 dimensions).
bash train_default_sfm.sh proprio_dims=9 \
  'datamodule.proprioception_dims.keep_indices=[[0,1],[3,7],[9,13]]'
```

`config_default_sfm_no_xy` retains `gripper,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz` in that order and sets both the dataset and model input width to 11. It starts a separate run from the same base FLOWER pretrained weights. Batch size (64 per GPU), GPUs 0/1, seed 42, 20 epochs, the 20,820-step LR schedule, 10-action chunks, F/T history, image transforms, and five-epoch checkpoint interval match the 13-D experiment. Delta X/Y action targets and X/Y velocities are retained. Dataset files and normalization statistics are shared without reprocessing.

`scripts/evaluate_default_sfm_trajectories.py` evaluates a completed checkpoint entirely on CPU in FP32 and produces a trajectory PDF, per-episode prediction arrays/CSVs, and metrics. By default it uses EMA weights and three evenly spaced demonstrations per task. It verifies all EMA keys/shapes and recreates the training-time frozen prompt using the original cached Florence constructor and seed. The first action of each 10-action prediction is decoded against that observation's recorded TCP pose; it is not cumulatively integrated. Only complete training-window starts are evaluated. Inference uses the stored image crops without color jitter, extra cropping, or repeated state/F/T normalization. This is training-data observation replay, not a robot rollout or held-out evaluation.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/jack/miniforge3/envs/flower/bin/python scripts/evaluate_default_sfm_trajectories.py \
  --run-dir logs/default_sfm/runs/flower_sfm_b64_20261028_043405 \
  --output reports/flower_sfm_cpu_trajectory_20261028 \
  --workers 9 --threads 4 --batch-size 8 --demos-per-task 3
```

Preprocessing and verification do not launch model training.

## Default SFM inference: FLOWER server and Kingston client

`server_flower_rpc.py` implements the existing Kingston client's `clean-share-policy-v2` HTTP/NPZ protocol. The original `Real_Robo.env_rpc_client` and its dependencies are unchanged. The protocol implementation is a pinned copy of Kingston's standalone reference server under `deployment/default_sfm/reference`; source hashes are recorded in `source_provenance.json`.

The prepared default checkpoint is `logs/default_sfm/runs/flower_sfm_b64_20261028_043405/seed_42/saved_models/epoch019.ckpt` (20 completed epochs, EMA weights, 13 state channels, 10 future actions). The server reads that run's saved `.hydra/config.yaml`, restores the original Florence constructor and training seed for the frozen prompt, and loads the checkpoint strictly. `--checkpoint` selects another checkpoint and its saved training config; `--training-config` is available if the config is elsewhere. State selection follows the saved config, including the 11-channel X/Y ablation. Existing training configs are not changed by inference setup.

The two machines have identical preprocessing bundles:

| Machine | Bundle path |
| --- | --- |
| FLOWER | `/home/jack/Skilled_FLOWER/datasets/default_sfm_crop_v2/rpc_bundle` |
| Kingston | `/home/csilab/share_projects/clean_share/team_codes/flower_default_sfm_rpc_20261028/rpc_bundle` |

Use the `rpc_bundle` directory for `--preprocess-bundle`. It adapts the dataset metadata into the existing client's schema and preserves the fitted statistics; no normalization is refitted. Its ID is `07c86390e0d1caacc8f9d0d15b14de5359c101eb1b46017fc29494da99eefd9a`. Both sides reject mismatched bundles. `scripts/build_default_sfm_rpc_bundle.py` creates a new version with `--output NEW_DIRECTORY` if needed and refuses to overwrite an existing bundle.

After reserving a free GPU, start the server on FLOWER. GPU 2 below is an example, not an availability claim:

```bash
cd /home/jack/Skilled_FLOWER
CUDA_VISIBLE_DEVICES=2 bash deployment/default_sfm/run_server.sh
```

The server requires an explicit `CUDA_VISIBLE_DEVICES`, defaults to port `45587`, and stays resident across episodes. CUDA inference uses FP16 autocast with FP32 parameters, matching training's mixed precision. Loading records a model audit at `logs/default_sfm_rpc/model_audit.json`. `bash deployment/default_sfm/run_server.sh --help` only displays arguments and does not load a model.

Once the server is ready, start the robot client on Kingston:

```bash
FLOWER_SERVER_IP=10.206.178.31 FLOWER_REQUEST_STEPS=5 \
  bash /home/csilab/share_projects/clean_share/team_codes/flower_default_sfm_rpc_20261028/run_client.sh usb
```

Replace `usb` with `audio_jack`, `barlatch`, `bnc`, `hdmi`, `knob`, `lan`, `outlet`, or `rotary_switch`. The wrapper supplies the exact training instruction from `rpc_bundle/instructions.json`, `--env_name default_sfm`, and the bundle path to the original client. It uses Kingston's existing `/home/csilab/miniforge3/envs/hilserl/bin/python`; `FLOWER_CLIENT_PYTHON` can override it. `FLOWER_SERVER_IP` defaults to `10.206.178.31`, the FLOWER address whose `/health` response was verified from Kingston. The originally supplied example address `115.145.172.57` refuses connections on port 45587. `FLOWER_SERVER_PORT`, `FLOWER_REQUEST_STEPS`, and `FLOWER_BUNDLE` are also configurable, and extra arguments are passed directly to the existing client.

The client retains its existing SpaceMouse intervention and monitor at `http://127.0.0.1:8125/` on Kingston. To view that monitor from a different machine, forward port 8125 through the existing SSH connection or run `ssh -L 8125:127.0.0.1:8125 Kingston` there.

The client crops and letterboxes both native RGB views to 224×224, wraps state roll/yaw, and normalizes state and the 50×6 F/T window. The server applies the saved image pixel transform and state selection once, then returns the full normalized checkpoint action horizon. The client restores physical base-frame delta translation/rotation and retains legacy binary gripper labels: closed −1, open +1. No model-side action denormalization or gripper relabeling occurs. The source bundle records the original 5-action preprocessing recipe; the protocol explicitly accepts a different inference horizon. The prepared checkpoint returns 10 actions, while `request_steps=5` controls the client's requery cadence, including latency compensation. A future 50-action checkpoint uses the same bundle without dropping its extra actions.

Preparation verification uses synthetic observations and tiny CPU stubs only. Six adapter tests passed. Kingston's actual existing preprocessor matched all three offline-training fixtures exactly for both images, state, and F/T; action inversion, gripper preservation, and wire serialization also passed. All 25 archived source hashes, including the original client, matched after preparation. This setup has not loaded the checkpoint, started an HTTP inference server, connected to robot hardware, or validated GPU inference/latency. The report is `deployment/default_sfm/client_bundle_check.json` locally and `client_bundle_check.json` in the Kingston deployment directory.

```bash
# FLOWER: CPU adapter checks, without constructing the trained model.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /home/jack/miniforge3/envs/flower/bin/python -m pytest -q tests/test_flower_rpc.py
# Kingston: existing preprocessing code only; no model, HTTP server, or robot.
/home/csilab/miniforge3/envs/hilserl/bin/python \
  /home/csilab/share_projects/clean_share/team_codes/flower_default_sfm_rpc_20261028/verify_client_bundle.py
```

## Original Repository

This repository is adapted from:

https://github.com/intuitive-robots/flower_vla_calvin

Please refer to the original repository for the base FLOWER VLA implementation.

# FlowerVLA

[Paper](https://www.arxiv.org/pdf/2509.04996), [Project Page](https://intuitive-robots.github.io/flower_vla/), [Pretraining Code](https://github.com/intuitive-robots/flower_vla_pret)

[Moritz Reuss](https://mbreuss.github.io/)<sup>1</sup>,
[Hongyi Zhou](https://hongyizhoucn.github.io/)<sup>1</sup>,
[Marcel Ruehle]()<sup>1</sup>,
[Ömer Erdinç Yağmurlu](https://scholar.google.com/citations?user=I_Mxp5cAAAAJ&hl=en)<sup>1</sup>,
[Fabian Otto](https://ottofabian.github.io/)<sup>2</sup>,
[Rudolf Lioutikov](http://rudolf.intuitive-robots.net/)<sup>1</sup>

<sup>1</sup>Intuitive Robots Lab (IRL), Karlsruhe Institute of Technology (KIT)
<sup>2</sup>Microsoft Research


##  FLOWER: An efficient & versatile Flow-VLA

FLOWER VLA is a lightweight, efficient Vision-Language-Action (VLA) policy for robotic manipulation tasks that achieves state-of-the-art performance on multiple benchmarks. Built on a rectified flow architecture with several key architecture features:

- **Efficient Architecture**: At ~1B parameters, FLOWER is significantly smaller than most VLA models
- **Low Training Cost**: Only requires ~200 GPU hours of pretraining
- **Low Memory Footprint**: Uses <3GB of GPU memory for inference with single image setting
- **SOTA Performance**: Achieves sota results on CALVIN and LIBERO benchmarks 

For the pretraining of FLOWER and finetuning for Aloha check out our other codebase.


## Model Overview

FLOWER VLA uses a Florence-2-large-based VLM combined with a rectified flow architecture to predict robot actions from visual observations and language instructions. 
The model efficiently handles multi-step action sequences through a chunking mechanism.

[Insert model architecture diagram here]

Key architectural components:
- Florence-2 DaVit Image Encoder [350M params]
- Language conditioning through cross-attention 
- Half of the Florence-2 LLM layers for vision and language fusion [205M]
- Rectified Flow Predition for fast action generation (all results in CALVIN and LIBERO are achieved using just 4 denoising steps) 
- Global AdaLN for parameter efficient conditioning
- Action chunking for multi-step action generation


## Installation
To begin, clone this repository locally
```bash
git clone --recurse-submodules git@github.com:intuitive-robots/flower_vla_calvin.git
export flower_calvin_ROOT=$(pwd)/flower_vla_calvin

```
Install requirements
(Note we provided a changed verison of pyhash, given numerous problems we encountered when installing it manually on our slurm cluster)
You can also try to install setup tools using pip. 
 
```bash
cd $flower_calvin_ROOT
conda create -n flower_cal python=3.9
conda activate flower_cal
cd calvin_env/tacto
pip install -e .
cd ..
pip install -e .
cd ..
cd LIBERO
pip install -r requirements.txt
pip install -e .
pip install numpy~=1.23
cd ..
pip install setuptools==57.5.0
cd pyhash-0.9.3
python setup.py build
python setup.py install
cd ..
```
Next we can install the rest of the missing packages

```
pip install -r requirements.txt
```

---

## Download
### CALVIN Dataset

If you want to train on the [CALVIN](https://github.com/mees/calvin) dataset, choose a split with:
```bash
cd $flower_calvin_ROOT/dataset
sh download_data.sh D | ABCD
```

## Training
To train the FLOWER FLOWERl with the 4 GPUS, run:
```
python flower/training.py 
```

You can use the pretrained FLOWER checkpoint from [hf-link](https://huggingface.co/mbreuss/flower_vla_pret) to train your own model on any of the datasets. 

Note that during training the full CALVIN eval or LIBERO rollouts will be called after _rollout_lh_skip_epochs_ and then every _callbacks.rollout_lh.rollout_freq_*1k training steps. Check out the training config for adopting the parameters.

For replication of the orginial training results I recommend to use 4 GPUs with a batch_size of 8 and train them for 40k steps for ABC (ABCD) and evaluating after 19 epochs to get the best possible results.
See training configs for details.

#### Preprocessing with CALVIN
Since FLOWER uses action chunking, it needs to load multiple (~10) `episode_{}.npz` files for each inference. In combination with batching, this results in a large disk bandwidth needed for each iteration (usually ~2000MB/iteration).
This has the potential of significantly reducing your GPU utilization rate during training depending on your hardware.
Therefore, you can use the script `extract_by_key.py` to extract the data into a single file, avoiding opening too many episode files when using the CALVIN dataset.

##### Usage example:
```shell
python preprocess/extract_by_key.py -i /YOUR/PATH/TO/CALVIN/ \
    --in_task all
```

```
python preprocess/extract_by_key.py -i /hkfs/work/workspace/scratch/ft4740-play3/data --in_task all
```

##### Params:
Run this command to see more detailed information:
```shell
python preprocess/extract_by_key.py -h
```

Important params:
* `--in_root`: `/YOUR/PATH/TO/CALVIN/`, e.g `/data3/geyuan/datasets/CALVIN/`
* `--extract_key`: A key of `dict(episode_xxx.npz)`, default is **'rel_actions'**, the saved file name depends on this (i.e `ep_{extract_key}.npy`)
Optional params:
* `--in_task`: default is **'all'**, meaning all task folders (e.g `task_ABCD_D/`) of CALVIN
* `--in_split`: default is **'all'**, meaning both `training/` and `validation/`
* `--out_dir`: optional, default is **'None'**, and will be converted to `{in_root}/{in_task}/{in_split}/extracted/`
* `--force`: whether to overwrite existing extracted data
Thanks to @ygtxr1997 for debugging the GPU utilization and providing a merge request.


## Evaluation

Download the pretrained FLOWER from Hugging Face: 
You can find all checkpoints under:

- [FLOWER Collection](https://huggingface.co/collections/mbreuss/flower-vla-67d60e95bf2990699fcef81f)

We provide pretrained checkpoints for all CALVIN and LIBERO challenges.

## Performance Comparison on CALVIN Challenges (1000 chains)

Below is the average performance of FLOWER on CALVIN. It currenty is SOTA for all variants of CALVIN:

| Train→Test | Method | PrT | 1 | 2 | 3 | 4 | 5 | **Avg. Len.** |
|------------|---------|-----|---|---|---|---|---|---------------|
| ABCD→D | Diff-P-CNN | × | 86.3% | 72.7% | 60.1% | 51.2% | 41.7% | 3.16±0.06 |
| | Diff-P-T | × | 78.3% | 53.9% | 33.8% | 20.4% | 11.3% | 1.98±0.09 |
| | RoboFlamingo | ✓ | 96.4% | 89.6% | 82.4% | 74.0% | 66.0% | 4.09±0.00 |
| | GR-1 | ✓ | 94.9% | 89.6% | 84.4% | 78.9% | 73.1% | 4.21±0.00 |
| | MDT | × | 98.6% | 95.8% | 91.6% | 86.2% | 80.1% | 4.52±0.02 |
| | MoDE | ✓ | 97.1% | 92.5% | 87.9% | 83.5% | 77.9% | 4.39±0.04 |
| | KomosVLA | ✓ | 98.0% | 93.6% | 85.4% | 77.8% | 70.4% | 4.49 |
| | **FLOWER (ours)** | × | **99.1%** | **97.8%** | **95.2%** | **92.4%** | **87.8%** | **4.67±0.04** |
| ABC→D | Diff-P-CNN | × | 63.5% | 35.3% | 19.4% | 10.7% | 6.4% | 1.35±0.05 |
| | Diff-P-T | × | 62.2% | 30.9% | 13.2% | 5.0% | 1.6% | 1.13±0.02 |
| | RoboFlamingo | ✓ | 82.4% | 61.9% | 46.6% | 33.1% | 23.5% | 2.47±0.00 |
| | GR-1 | ✓ | 85.4% | 71.2% | 59.6% | 49.7% | 40.1% | 3.06±0.00 |
| | 3DDA | × | 93.8% | 80.3% | 66.2% | 53.3% | 41.2% | 3.35 |
| | MoDE | ✓ | 96.2% | 88.9% | 81.1% | 71.8% | 63.5% | 4.01±0.04 |
| | KomosVLA | ✓ | 98.0% | 93.6% | 85.4% | 77.8% | 70.4% | 4.25 |
| | VPP | ✓ | 95.7% | 91.2% | 86.3% | 81.0% | 75.0% | 4.29 |
| | Seer | ✓ | 96.3% | 91.6% | 86.1% | 80.3% | 74.0% | 4.28 |
| | **FLOWER (ours)** | × | **99.3%** | **95.9%** | **90.5%** | **84.8%** | **77.5%** | **4.54±0.02** |
| D→D| **FLOWER (ours)** | × | **98.4%** | **94.0%** | **87.9%** | **81.7%** | **74.1%** | **4.36±0.04** |

## Performance on LIBERO Benchmarks

FLOWER achieves strong performance across all LIBERO benchmarks:

| Benchmark | FLOWER Success Rate |
|-----------|---------------------|
| LIBERO-10 | 94.5% |
| LIBERO-90 | 93.4% | 
| LIBERO-SPATIAL | 97.2% |
| LIBERO-OBJECT | 99.3% | 
| LIBERO-GOAL | 96.9% | 


#### Common Issues

Sometimes this causes problems for the python env so just delete it:

```python
log.info(f"Using calvin_env with commit {get_git_commit_hash(Path(calvin_env.__file__))}.")
```
The path for this line is in the CALVIN env repo: https://github.com/mees/calvin_env/blob/797142c588c21e76717268b7b430958dbd13bf48/calvin_env/envs/play_table_env.py#L72

---

## Acknowledgements

This work is only possible because of the code from the following open-source projects and datasets. We thank all authors for their work:

#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)

License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### LIBERO

Original: [https://github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

License: [https://github.com/Lifelong-Robot-Learning/LIBERO?tab=MIT-1-ov-file](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=MIT-1-ov-file)

#### Mimictest 

Original: [mimictest](https://github.com/EDiRobotics/mimictest)
License: [license](https://github.com/EDiRobotics/mimictest?tab=Apache-2.0-1-ov-file)

#### HULC
Original: [https://github.com/lukashermann/hulc](https://github.com/lukashermann/hulc)

License: [MIT](https://github.com/lukashermann/hulc/blob/main/LICENSE)

#### FLOWER Pretraining Codebase

Original: [https://github.com/intuitive-robots/FLOWER_Diffusion_Policy](https://github.com/intuitive-robots/FLOWER_Diffusion_Policy)

License: [https://github.com/intuitive-robots/FLOWER_Diffusion_Policy/blob/main/LICENSE](https://github.com/intuitive-robots/FLOWER_Diffusion_Policy/blob/main/LICENSE) 


## Citation

If you found the code usefull, please cite our work: (arxiv coming very soon)

```bibtex
@inproceedings{
reuss2025flower,
title={{FLOWER}: Democratizing Generalist Robot Policies with Efficient Vision-Language-Flow Models},
author={Moritz Reuss and Hongyi Zhou and Marcel R{\"u}hle and {\"O}mer Erdin{\c{c}} Ya{\u{g}}murlu and Fabian Otto and Rudolf Lioutikov},
booktitle={9th Annual Conference on Robot Learning},
year={2025},
url={https://openreview.net/forum?id=JeppaebLRD}
}
```
