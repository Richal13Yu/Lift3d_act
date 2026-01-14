python -m lift3d.tools.act_policy_debug \
  --config-name train_recover \
  hydra.run.dir=/projects/surgical-video-digital-twin/new_results/act_peg_recover/overfit_first2eps_act_10000_lr1e-5_kl0 \
  benchmark=act_offline \
  agent=lift3d_act \
  task_name=peg_recover \
  dataset_dir=/projects/surgical-video-digital-twin/datasets/act_peg_recover/1216/zarr_first2eps \
  dataloader.batch_size=16 dataloader.num_workers=8 dataloader.shuffle=true \
  train.num_epochs=10000 train.learning_rate=1e-5 train.kl_weight=0 \
  evaluation.validation_frequency_epochs=100 evaluation.validation_trajs_num=1 \
  wandb.mode=online \
  +debug.enable=true +debug.log_every_iters=2500 +debug.dump_first_n_iters=5 \
  +debug.mirror_dir=/projects/surgical-video-digital-twin/new_results/act_peg_recover/overfit_first2eps_act_act_10000_lr1e-5_k0