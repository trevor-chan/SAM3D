import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import sys
import utils
import math
import time
import tqdm
import argparse
from segment_anything import sam_model_registry, SamPredictor
import open3d as o3d

import geometry
import scale_transform
import platonics
import segmentfunction
import prompting





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the image directory")
    parser.add_argument("-r", "--rotations", help="type of rotations to apply: ['ortho','cubic','ico','dodeca']", default="ico")
    parser.add_argument("-s", "--slices", help="number of slices for segmentation inference along each axis", default=100)
    parser.add_argument("-o", "--outdir", help="location to save the final mask")
    parser.add_argument("-ch", "--checkpoint", help="location of the SAM model checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    # start timer:
    starttime = time.time()

    # get list of transforms:
    if args.rotations == "ortho":
        transform_list = platonics.get_ortho_transforms()
    elif args.rotations == "cubic":
        transform_list = platonics.get_cubic_transforms()
    elif args.rotations == "ico":
        transform_list = platonics.get_icosahedron_transforms()
    elif args.rotations == "dodeca":
        transform_list = platonics.get_dodecahedron_transforms()
    else:
        print("rotations must be supplied in the form of ['ortho','cubic','ico','dodeca']")
        return 0
    
    # open image and get slices
    with open(args.path, 'r') as f:
        image = utils.padtocube(utils.load3dmatrix(f))

    # make a temporary directory to save the slices
    tempdir = "tempdir"
    utils.mkdir(tempdir)
    slices_list = scale_transform.get_prompt_slices(image, tempdir, transform_list)
    
    #call prompting script
    prompting.main(tempdir)
    
    # parse prompts
    pos_seg, neg_seg = scale_transform.parse_prompts(tempdir, slices_list)
    
    # initialize SAM model
    sam_checkpoint = args.checkpoint
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, device=device)
    predictor = SamPredictor(sam)
    
    # segmentation inference
    prompting_slices_dict = dict()
    count = 0
    allpoints = []
    boundarypoints = []
    
    z_coord_list = np.linspace(-1,1,args.slices+2)[1:-1]
    for t in tqdm.tqdm(transform_list):
        transformed_img = scale_transform.global_to_local(image, t)
        matrix_shape = np.array(transformed_img.shape)
        
        for z in z_coord_list:
            zidx = int((z+1)/2*matrix_shape[2])
            slice_transformed_img = transformed_img[:,:,zidx]
            slice_shape = slice_transformed_img.shape
            
            pos_intersections, neg_intersections = scale_transform.get_intersections(matrix_shape, pos_seg, neg_seg, t, z)
            
            pos_intersections = [[pt[1],pt[0]] for pt in pos_intersections]
            neg_intersections = [[pt[1],pt[0]] for pt in neg_intersections]
            
            if len(pos_intersections) != 0:
                prompt = [pos_intersections, neg_intersections]
                points, boundary = segmentfunction.segment(predictor, scale_transform.normalize(slice_transformed_img), prompt)
                
                point_indices = np.nonzero(points)
                point_indices = np.array(zip(point_indices[0],point_indices[1]))
                
                for p in point_indices:
                    p = scale_transform.index_to_coord(p, t, matrix_shape)
                    allpoints.append(p)
    
    # create pointcloud
                
    





if __name__ == "__main__":
    main()