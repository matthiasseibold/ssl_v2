#!/bin/bash

scenes="1 2 3 4 6 7 8 9 10 11 12 13 14 15 16 17 18 19 21 22"

#for res in 6 4 3 2 1; do
for res in ""; do
  thres=$(( 200 / $res / $res ))
  echo "__ Input Resolution Downscaling Test s=1/$res t=$thres __"
  for scene in $scenes; do
    python3 localization3d/bbox_estimation.py --scene_id $scene --render_to_file --event_frames data/prediction/event_localization2d/ --only_event_frames --dbscan_min_weight $thres --rgbd_downsampling_factor $res;
  done
done

#for thres in 800 400 300 200 150 100 50; do
for thres in 300 150 100 50; do
  echo "__ DBSCAN minWeight Test t=$thres __"
  for scene in $scenes; do
    python3 localization3d/bbox_estimation.py --scene_id $scene --render_to_file --event_frames data/prediction/event_localization2d/ --only_event_frames --dbscan_min_weight $thres --rgbd_downsampling_factor 1;
  done
done

#for radius in 7.5 15 22.5 30 45 60 120; do
for radius in 7.5 15 22.5 45 60 120; do
  echo "__ DBSCAN Radius Test r=$radius __"
  for scene in $scenes; do
    python3 localization3d/bbox_estimation.py --scene_id $scene --render_to_file --event_frames data/prediction/event_localization2d/ --only_event_frames --dbscan_radius $radius;
  done
done

