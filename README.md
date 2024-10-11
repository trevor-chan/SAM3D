# Project repository for SAM3D: Zero-Shot Semi-Automatic Segmentation in 3D Medical Images with the Segment Anything Model

This is a work in progress repository for code related to the paper "SAM3D: Zero-Shot Semi-Automatic Segmentation in 3D Medical Images with the Segment Anything Model"

If you decide to use or adapt this code, please cite the arxiv paper at: https://doi.org/10.48550/arXiv.2405.06786

## Installation

Major dependencies for the code include pytorch, segment-anything, and open3d-python. For the UI to work, you may also need to install PyQt5. Minor dependencies may also be necessary and can be installed via pip or conda.

An Nvidia GPU is recommended for fast inference. If you are running on a computer without a GPU, you can run the model on a CPU but we recommend using fewer slices to save on inference time. You may also need to adjust the default values for mask post-processing.

## To run code

The main script is sam3d.py. This script accepts a number of arguments, which can be seen by running $python sam3d.py --help

If running with the defaults, the only argument that needs to be supplied is --path / -p, the path to a stack of 2D images for the volume to be segmented.

## Contact

Please reach out to tjchan@seas.upenn.edu for questions about installing or running code. Please also keep in mind that this is a work in progress, and changes may be made in the future.
