import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


filepath = r"C:\\Users\\aarus\\Downloads\\OrganSegmentations\\00002_z0000.png"
checkpointfilepath = r"C:\\Users\\aarus\\Downloads\\sam_vit_h_4b8939.pth"
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 


image = cv2.imread(filepath)
print(image.shape)


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


sam_checkpoint = checkpointfilepath
model_type = "vit_h"

# device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image(image)

def createoriginalpoints(movingnp):
    print("please input points")
    posoriginalpoints = []
    negoriginalpoints = []
    def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
    
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            posoriginalpoints.append((x,y))
            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + ',' +
                        str(y) + '+', (x,y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image', img)
    
        # checking for right mouse clicks     
        if event==cv2.EVENT_RBUTTONDOWN:
    
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            negoriginalpoints.append((x,y))
            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(img, str(x) + ',' +
                        str(y) + '-',
                        (x,y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image', img)
    
    # driver function
    if __name__=="__main__":
    
        # reading the image
        img = movingnp
    
        # displaying the image
        cv2.imshow('image', movingnp)
    
        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', click_event)
    
        # wait for a key to be pressed to exit
        cv2.waitKey(0)
    
        # close the window
        cv2.destroyAllWindows()

    posoriginalpointsarray = np.array(posoriginalpoints, dtype=np.float32)
    negorignalpointsarray = np.array(negoriginalpoints, dtype=np.float32)

    return posoriginalpointsarray, negorignalpointsarray

points = createoriginalpoints(image)
input_point = np.concatenate([points[0], points[1]], axis=0)

input_labelpos = np.ones(len(points[0]), dtype=np.int64)
input_labelneg = np.zeros(len(points[1]), dtype=np.int64)
# for i in range(0, len(input_point[0])):
#     input_labelpos[i] = 1

# for i in range(0, len(input_point[1])):
#     input_labelneg[i] = 0

input_label = np.concatenate([input_labelpos, input_labelneg], axis=0)

print(input_point)
print(input_label)

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
final_mask = masks.astype(np.uint8)
print(final_mask.shape)
print(final_mask)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()

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