import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def segment(predictor,image, zidx, promptlists):
    predictor.set_image(image)
    # print("done2")
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

    # print(input_point)
    # print(input_label)

    # print("done4")
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

    # print("done5")
    final_mask = masks.astype(np.uint8)
    # print(final_mask.shape)
    # print(final_mask)

    # Remove the singleton dimension from final_mask
    final_mask_squeezed = np.squeeze(final_mask)

    # Now plot the final mask
    # plt.figure(figsize=(10, 10))
    # plt.imshow(final_mask_squeezed, cmap='gray')
    # plt.axis('off')
    # plt.title('Segmentation Mask')
    # plt.show()

    from scipy.ndimage import binary_erosion

    # Example segmentation mask


    # Erode the mask

    eroded_mask = binary_erosion(final_mask_squeezed)

    # Find the outer boundary by subtracting the eroded mask from the original mask
    outer_boundary = final_mask_squeezed - eroded_mask


    for i in pospoints:
        plt.scatter(i[0], i[1], c='g', s=10)
    for i in negpoints:
        plt.scatter(i[0], i[1], c='r', s=10)
    print("max image = ", np.max(image[:,:,0]))
    print("min image = ", np.min(image[:,:,0]))
    print("max boundary = ", np.max(outer_boundary))
    print("min boundary = ", np.min(outer_boundary))
    print("max points = ", np.max(pospoints))
    print("min points = ", np.min(pospoints))

    plt.imshow(image[:,:,0], cmap='gray')
    plt.imshow(outer_boundary, cmap='inferno', alpha=0.5)
    plt.show()
    return final_mask_squeezed, outer_boundary


print("done")

