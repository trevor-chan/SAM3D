import numpy as np
from PIL import Image
import os
import math


def load3dmatrix(folder):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
    filepaths.sort()
    images = [Image.open(f) for f in filepaths]
    image= np.stack([np.array(im) for im in images])

    return image

folder = r"C:\Users\aarus\Downloads\slices_for_prompting\I1351301"

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