import numpy as np
from PIL import Image
import os

def load3dmatrix(folder):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
    filepaths.sort()
    images = [Image.open(f) for f in filepaths]
    matrix= np.stack([np.array(im) for im in images])
    return matrix

folder = "C:\\Users\\aarus\\Downloads\\slices_for_prompting"
