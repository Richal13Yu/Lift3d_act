
export PYTHONPATH=$PWD/third_party:$PYTHONPATH
python -m lift3d.tools.act_policy \
  --config-name=train_recover \
  benchmark=act_offline \
  agent=lift3d_act \
  task_name=peg_recover \
  dataloader.batch_size=16 \
  dataset_dir=/projects/surgical-video-digital-twin/datasets/act_peg_recover/1216/zarr \
  hydra.run.dir=/projects/surgical-video-digital-twin/new_results/act_peg_recover/lift3d