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


