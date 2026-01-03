python lift3d/tools/convert_to_zarr.py \
  --root /projects/surgical-video-digital-twin/datasets/act_peg_recover/1104/datasets \
  --output-zarr /projects/surgical-video-digital-twin/datasets/act_peg_recover/1216/zarr \
  --registration-json /projects/surgical-video-digital-twin/datasets/act_peg_recover/1104/handeye/PSM1-registration-open-cv.json \
  --action-type hybrid_relative \
  --crop 585 200 1205 880 \
  --pc-size 640 480 \
  --img-size 224 \
  --num-points 1024 \
  --depth-scale 0.001