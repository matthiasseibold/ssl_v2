# Sound Source Localization for 3D Spatial Mapping of Surgical Actions in Dynamic Scenes

Source code for the IPCAI2026 publication "Sound Source Localization for 3D Spatial Mapping of Surgical Actions in Dynamic Scenes" ([arXiv link](https://arxiv.org/abs/2510.24332)).

## Dependencies
We rely on StereoLabs [ZED SDK](https://www.stereolabs.com/en-ch/developers/release) to read the captured stereo-RGB videos.
Make sure to install this first. Run `python3 /usr/local/zed/get_python_api.py` to get the Python API wrapper.

Use `uv sync` to install all other dependencies.

## Dataset
The dataset can be downloaded at: https://doi.org/10.5281/zenodo.18682076
The scripts in this repository expect the dataset to be located at `data/dataset/` by default.


## Event Detection
The implementation of our event detection stage is located under `eventdetection/ast`.

## Sound Source Localization
The implementation of our sound source locazation stage is located under `localization3d`.

## Demo Video
[Watch with sound 🔊]

https://github.com/user-attachments/assets/a1ffa828-e558-432a-bb68-4746b4cbaad5

