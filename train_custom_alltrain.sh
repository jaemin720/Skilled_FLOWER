CUDA_VISIBLE_DEVICES=2,3 python flower/training_calvin.py --config-name config_custom_calvin_alltrain_skillvae \
  root_data_dir=/home/jack/flower_vla_calvin/multi_task_dataset_raw_action_alltrain \
  batch_size=16 \
  num_workers=4 \
  max_epochs=10 \
  devices=2