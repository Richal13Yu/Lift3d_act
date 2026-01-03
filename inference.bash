python lift3d_act_infer_dvrk_ros1.py \
  --hydra_config /path/to/your_run/.hydra/config.yaml \
  --ckpt /path/to/your_run/best_model.pth \
  --rgb_topic /rgb/image_rect_color \
  --depth_topic /depth_to_rgb/hw_registered/image_rect/ \
  --caminfo_topic /rgb/camera_info \
  --registration_json /home/xxx/PSM1-registration-open-cv.json \
  --motion_mode hybrid \
  --pos_scale 0.001 \
  --exec_horizon 50