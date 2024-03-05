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
import os
import torch
import shutil

import geometry
import scale_transform
import platonics
import segmentfunction
import prompting
import recomposition





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the image directory")
    parser.add_argument("-r", "--rotations", help="type of rotations to apply: ['ortho','cubic','ico','dodeca']", default="ico")
    parser.add_argument("-s", "--slices", help="number of slices for segmentation inference along each axis", default=100)
    parser.add_argument("-o", "--outdir", help="location to save the final mask", default="outputs")
    parser.add_argument("-ch", "--checkpoint", help="location of the SAM model checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--reslice", help="if false, skip the initial reslicing step", default=1)
    parser.add_argument("--reprompt", help="if false, skip the initial prompting step", default=1)
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
    print('transforms made')
    
    # open image and get slices
    image = utils.padtocube(utils.load3dmatrix(args.path))
    print('image loaded')

    # make a temporary directory to save the slices
    tempdir = "tempdir"
    if int(args.reslice):
        shutil.rmtree(tempdir)
        os.makedirs(tempdir)
        slices_list, transformed_arrays = scale_transform.get_prompt_slices(image, tempdir, transform_list)
    else:
        slices_list, transformed_arrays = scale_transform.get_prompt_slices(image, tempdir, transform_list, reslice=False)
    
    #call prompting script
    if int(args.reprompt):
        prompting.main(tempdir)
    
    # parse prompts
    pos_seg, neg_seg = scale_transform.parse_prompts(tempdir, slices_list)
    
    # initialize SAM model
    sam_checkpoint = args.checkpoint
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    print('model loaded')
    
    # segmentation inference
    prompting_slices_dict = dict()
    count = 0
    allpoints = []
    boundarypoints = []
    
    z_coord_list = np.linspace(-1,1,int(args.slices)+2)[1:-1]
    for rotnum, t in tqdm.tqdm(enumerate(transform_list)):
        if len(transformed_arrays) == 0:
            transformed_img = scale_transform.global_to_local(image, t)
        else:
            transformed_img = transformed_arrays[rotnum]
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
                
                undersample = matrix_shape[0]//int(args.slices)
                undersampled_points = np.zeros_like(points)
                undersampled_points[::undersample, ::undersample] = points[::undersample, ::undersample]
                
                point_indices = np.nonzero(undersampled_points)
                point_indices = np.stack((point_indices[0],point_indices[1]), axis=1)
                
                newpoints = [scale_transform.index_to_coord(np.concatenate((point_indices[i], [zidx])), t, matrix_shape) for i in range(point_indices.shape[0])]
                allpoints.append(newpoints)
    

    # pointcloud refinement loop
    print('point cloud refinement')
    running = True
    valid_inputs = ['evaluate', 'downsample', 'outliers', 'done']
    downsample, outliers, n_neighbors, std_ratio = 5, 1, 20, 0.5  # Set default values
    pcd = recomposition.create_point_cloud(allpoints, visualize=True, downsample=downsample, outliers=outliers, n_neighbors=n_neighbors, std_ratio=std_ratio)
    while running:
        user_input = input("Enter a command (evaluate, downsample, outliers, done): ").lower()  # Convert input to lowercase
        # Check if the input is valid
        if user_input in valid_inputs:
            # Perform actions based on user input
            if user_input == 'downsample':
                downsample_input = input(f"Current downsample = {downsample}, enter a new value: ").lower()
                assert int(downsample_input) > 0, "choose an integer factor greater than 0."
                downsample = int(downsample_input)
            elif user_input == 'outliers':
                n_neighbors_input = input(f"Current n_neighbors = {n_neighbors}, enter a new n_neighbors: ").lower()
                std_ratio_input = input(f"Current std_ratio = {std_ratio}, enter a new std_ratio: ").lower()
                assert int(n_neighbors_input) >= 0, "n_neighbors must be an integer greater than or equal to 0."
                assert float(std_ratio_input) >= 0, "std_ratio must be a float greater than or equal to 0."
                n_neighbors = int(n_neighbors_input)
                std_ratio = float(std_ratio_input)
                if n_neighbors == 0 or std_ratio == 0:
                    outliers = 0
                else:
                    outliers = 1
            elif user_input == 'done':
                running = False
            elif user_input == 'evaluate':
                pcd = recomposition.create_point_cloud(allpoints, visualize=True, downsample=downsample, outliers=outliers, n_neighbors=n_neighbors, std_ratio=std_ratio)
        else:
            print("Invalid input. Please enter one of (voxsize, resolution, dilation, erosion, fillholes), (evaluate), or (done) if finished.")

    
    
    # refinement loop
    print('voxel mask refinement')
    valid_inputs = ['evaluate', 'voxsize', 'resolution', 'dilation', 'erosion', 'fillholes', 'done']
    running = True
    voxsize, resolution, dilation, erosion, fillholes = 0.0078, image.shape[0], 5, 5, True  # Set default values
    mask = recomposition.voxel_density_mask(pcd, vox_size = voxsize, resolution = resolution, dilation = dilation, erosion = erosion, fill_holes = fillholes)
    recomposition.draw_orthoplanes(image, mask)
    
    while running:
        user_input = input("Enter a command (evaluate, voxsize, resolution, dilation, erosion, fillholes, done): ").lower()  # Convert input to lowercase
        if user_input in valid_inputs:
            # Perform actions based on user input
            if user_input == 'voxsize':
                vox_size_input = input(f"Current voxel size = {voxsize}, enter a new voxel size: ").lower()
                assert float(vox_size_input) > 0 and float(vox_size_input) < 1, "Voxel size must be greater than 0 and less than 1."
                voxsize = float(vox_size_input)
            elif user_input == 'resolution':
                resolution_input = input(f"Current resolution = {resolution}, enter a new resolution: ").lower()
                assert int(resolution_input) > 0, "Resolution must be an integer greater than 0."
                resolution = int(resolution_input)
            elif user_input == 'dilation':
                dilation_input = input(f"Current dilation = {dilation}, enter a new dilation: ").lower()
                assert int(dilation_input) >= 0, "dilation must be an integer greater than or equal to 0."
                dilation = int(dilation_input)
            elif user_input == 'erosion':
                erosion_input = input(f"Current erosion = {erosion}, enter a new erosion: ").lower()
                assert int(erosion_input) >= 0, "erosion must be an integer greater than or equal to 0."
                erosion = int(erosion_input)
            elif user_input == 'fillholes':
                if fillholes:
                    print(f"Fill holes was set to True. Fill holes is now False.")
                    fillholes = False
                else:
                    print(f"Fill holes was set to False. Fill holes is now True.")
                    fillholes = True
            elif user_input == 'done':
                running = False
            elif user_input == 'evaluate':
                mask = recomposition.voxel_density_mask(pcd, vox_size = voxsize, resolution = resolution, dilation = dilation, erosion = erosion, fill_holes = fillholes)
                recomposition.draw_orthoplanes(image, mask)
                print("Mask created.")
        else:
            print("Invalid input. Please enter one of (voxsize, resolution, dilation, erosion, fillholes), (evaluate), or (done) if finished.")
            
    # save mask
    savename = input("name to save w/out extension: ")
    if len(args.outdir) == 0:
        args.outdir = "."
    utils.save_mrc(image, f'{args.outdir}/{savename}_image.mrc')
    utils.save_mrc(mask, f'{args.outdir}/{savename}_mask.mrc')
    
    elapsed = int(time.time()-starttime)
    print(f"Total time elapsed: {elapsed//60} minutes, {elapsed%60} seconds.")
        

if __name__ == "__main__":
    main()