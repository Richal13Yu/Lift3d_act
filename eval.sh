python eval.py \
  --config-path /projects/surgical-video-digital-twin/new_results/act_peg_recover/3dact_lr1e-5_kl0/.hydra \
  --config-name config \
  hydra.run.dir=/projects/surgical-video-digital-twin/new_results/act_peg_recover/3dact_lr1e-5_kl0/${now:%Y-%m-%d}/${now:%H-%M-%S} \
  ++evaluation.checkpoint_path=/projects/surgical-video-digital-twin/new_results/act_peg_recover/3dact_lr1e-5_kl0/best_model.pth \
  ++evaluation.split=validation \
  ++evaluation.episode_id=0 \