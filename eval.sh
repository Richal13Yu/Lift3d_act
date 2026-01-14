python eval.py \
  --config-path /projects/surgical-video-digital-twin/new_results/act_peg_recover/overfit_first2eps_act_10000_lr1e-5_kl0/.hydra \
  --config-name config \
  hydra.run.dir=/projects/surgical-video-digital-twin/new_results/act_peg_recover/overfit_first2eps_act_10000_lr1e-5_kl0/${now:%Y-%m-%d}/${now:%H-%M-%S} \
  ++evaluation.checkpoint_path=/projects/surgical-video-digital-twin/new_results/act_peg_recover/overfit_first2eps_act_10000_lr1e-5_kl0/best_model.pth \
  ++evaluation.split=train \
  ++evaluation.episode_id=0 \