export CUDA_VISIBLE_DEVICES=3
python server_flower.py \
  --checkpoint_path "/data2/jack/log/runs/2026-05-10/18-24-49_seed242/seed_242/saved_models/epoch=19_train/action_loss=0.0216.ckpt" \
  --dataset_root /home/jack/flower_vla_calvin/multi_task_dataset_raw_action_alltrain \
  --host 0.0.0.0 \
  --port xxxx
