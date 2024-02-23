import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

filepath = r"C:\\Users\\aarus\\Downloads\\OrganSegmentations\\00002_z0000.png"
image = cv2.imread(filepath)
checkpointfilepath = r"C:\\Users\\aarus\\Downloads\\sam_vit_h_4b8939.pth"
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


sam_checkpoint = checkpointfilepath
model_type = "vit_h"

# device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

predictor = SamPredictor(sam)

print("done")

def segment(image, zidx, promptlists):
    predictor.set_image(image)
    print("done2")
    pospoints = promptlists[0]
    negpoints = promptlists[1]

    input_point = np.concatenate([pospoints, negpoints], axis=0)

    input_labelpos = np.ones(len(pospoints), dtype=np.int64)
    input_labelneg = np.zeros(len(negpoints), dtype=np.int64)
# for i in range(0, len(input_point[0])):
#     input_labelpos[i] = 1

# for i in range(0, len(input_point[1])):
#     input_labelneg[i] = 0

    input_label = np.concatenate([input_labelpos, input_labelneg], axis=0)

    print(input_point)
    print(input_label)

    print("done4")
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    print("done5")
    final_mask = masks.astype(np.uint8)
    print(final_mask.shape)
    print(final_mask)

    # Remove the singleton dimension from final_mask
    final_mask_squeezed = np.squeeze(final_mask)

    # Now plot the final mask
    plt.figure(figsize=(10, 10))
    plt.imshow(final_mask_squeezed, cmap='gray')
    plt.axis('off')
    plt.title('Segmentation Mask')
    plt.show()

    from scipy.ndimage import binary_erosion

    # Example segmentation mask


    # Erode the mask

    eroded_mask = binary_erosion(final_mask_squeezed)

    # Find the outer boundary by subtracting the eroded mask from the original mask
    outer_boundary = final_mask_squeezed - eroded_mask

    plt.imshow(outer_boundary, cmap='gray')
    plt.show()


#take the final mask filled in
#stack the masks from all slices
#set threshold based on density
#run gaussian kernel to determine how close points are
    
segment(image, 0, [[(0,1)], [(1,0)]])