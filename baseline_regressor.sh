#!/bin/bash

scenes="1 2 3 4 6 7 8 9 10 11 12 13 14 15 16 17 18 19 21 22"

for scene in $scenes; do
  python3 localization3d/weighted_average_baseline.py --scene_id $scene --event_frames data/prediction/event_localization2d/ --only_event_frames ;
done