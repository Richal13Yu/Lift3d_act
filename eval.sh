python eval.py \
  --config-path /projects/surgical-video-digital-twin/new_results/act_peg_recover/lift3d/.hydra \
  --config-name config \
  hydra.run.dir=/projects/surgical-video-digital-twin/new_results/act_peg_recover/lift3d_eval/${now:%Y-%m-%d}/${now:%H-%M-%S} \
  +evaluation.checkpoint_path=/projects/surgical-video-digital-twin/new_results/act_peg_recover/lift3d/best_model.pth \
  +evaluation.split=validation \
  +evaluation.plot_index=0