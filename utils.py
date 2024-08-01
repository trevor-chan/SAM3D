import numpy as np
from PIL import Image
import os
import math
import mrcfile
import pydicom as dicom
import matplotlib.pyplot as plt


def load3dmatrix(folder, datatype):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.%s' % datatype)]
    # print(dicom.dcmread(filepaths[0]).pixel_array)
    filepaths.sort()
    if datatype == "png":
        images = [Image.open(f) for f in filepaths]
    if datatype == "dcm":
        images = [dicom.dcmread(f).pixel_array for f in filepaths]
    image = np.stack(images, axis=-1)
    image = (image - np.amin(image)) / (np.amax(image) - np.amin(image)) * 255
    return image

def padtocube(array):
    shape = array.shape
    max_dim = max(shape)
    left_pad1 = (max_dim - shape[0]) // 2
    right_pad1 = max_dim - shape[0] -  left_pad1
    left_pad2 = (max_dim - shape[1]) // 2
    right_pad2 = max_dim - shape[1] -  left_pad2
    left_pad3 = (max_dim - shape[2]) // 2
    right_pad3 = max_dim - shape[2] -  left_pad3
    padded_array = np.pad(array, ((left_pad1, right_pad1), (left_pad2, right_pad2), (left_pad3, right_pad3)), mode='constant', constant_values=0)
    return padded_array

def save_mrc(array, filepath):
    with mrcfile.new(filepath, overwrite=True) as mrc:
        mrc.set_data(array.astype(np.float32))

def load_mrc(filepath):
    with mrcfile.open(filepath) as mrc:
        mrc_data = mrc.data
        array = np.array(mrc_data)
    return array