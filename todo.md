# Step 1 Get Prompt Points

- directory of jpg/png images
- for each images, save a dictionary of prompt points (pos, neg)
    [
        {
            "image_name": "path_2_img_1.png",
            "pos": 
                [
                    [(x0,y0), (x1,y1), ...],
                    [(x0,y0), (x1,y1), ...],
                ]
            ,
            "neg": 
                [
                    [(x0,y0), (x1,y1), ...],
                    [(x0,y0), (x1,y1), ...],
                ]
                    
        },...
    ]
    save that structure in a json file
- rotation matrix


# get test matrix

# get the list of transforms

# slice the matrix using global_to_local function. Do it to every transform.

# Add prompting:
    # save slices to image files (naming convention: 001.png)
    # open image files and save prompt points to jason 

# convert prompt points to zero_centric coordinates using index_to_coord

# for each transform:
    # calculate the rotated array
    # for n slices:
        # get the slice of the rotated array,
        # calculate the prompts intersecting with that slice
        # feed into SAM inference function
        # Get mask points for that slice
        # Convert to global coord, append to list of global mask points
        
# take global mask points, convert to volumetric mask
