python -m lift3d.tools.train_policy \
  --config-name=train_recover \
  agent=lift3d_act \
  task_name=peg_recover \
  camera_name=rgb \
  dataloader.batch_size=16 \
  dataset_dir=/projects/surgical-video-digital-twin/datasets/act_peg_recover/1216/zarr