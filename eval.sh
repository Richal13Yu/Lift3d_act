python eval.py \
  --config-path /projects/surgical-video-digital-twin/new_results/act_peg_recover/lift3d_0108/.hydra \
  --config-name config \
  hydra.run.dir=/projects/surgical-video-digital-twin/new_results/act_peg_recover/lift3d_eval_0108/${now:%Y-%m-%d}/${now:%H-%M-%S} \
  +evaluation.checkpoint_path=/projects/surgical-video-digital-twin/new_results/act_peg_recover/lift3d_0108/best_model.pth \
  +evaluation.split=validation \
  +evaluation.episode_id=0 \
  +evaluation.episode_length=700 \
  +evaluation.eval_batch_size=16