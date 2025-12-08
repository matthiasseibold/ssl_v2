# Sound Source Localization for 3D Spatial Mapping of Surgical Actions in Dynamic Scenes

Source code for the IPCAI2026 submission "Sound Source Localization for 3D Spatial Mapping of Surgical Actions in Dynamic Scenes".

## Dependencies
We rely on StereoLabs [ZED SDK](https://www.stereolabs.com/en-ch/developers/release) to read the captured stereo-RGB videos.
Make sure to install this first. Run `python3 /usr/local/zed/get_python_api.py` to get the Python API wrapper.

Use `uv sync` to install all other dependencies.

## Dataset
The dataset will be made available here soon.

## Event Detection
The implementation of our event detection stage is located under `eventdetection/ast`.

## Sound Source Localization
The implementation of our sound source locazation stage is located under `localization3d`.