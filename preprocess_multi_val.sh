python /NHNHOME/WORKSPACE/0226010443_A/jack/flower_original/origin/FLOWER_Calvin_Custom/preprocess/convert_multitask_raw_pkl_to_flower_only_trim_valdir_taskmap.py \
  --input /NHNHOME/WORKSPACE/0226010443_A/jack/data/multi \
  --val_input /NHNHOME/WORKSPACE/0226010443_A/jack/data/multi_val \
  --output_root /NHNHOME/WORKSPACE/0226010443_A/jack/data/flower_dataset/multi_task_dataset_with_val \
  --pkl_glob "**/*.pkl" \
  --include_right_ft \
  --include_right_force_history \
  --action_source raw \
  --zero_action_trailing_keep 5 \
  --overwrite
