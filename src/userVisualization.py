import numpy as np
import cv2

def applyMaskToImage(image, binary_mask, color_for_masked_region=[0, 0, 255]):
    # Convert the binary mask to a 3-channel image
    binary_mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Ensure that color_for_masked_region is a NumPy array with dtype=np.uint8
    color_for_masked_region = np.array(color_for_masked_region, dtype=np.uint8)

    # Use the binary mask to create a mask for the colored region
    masked_region = cv2.bitwise_and(binary_mask_color, color_for_masked_region)

    # Combine the masked region and the original image using bitwise_or
    result_image = cv2.bitwise_or(image, masked_region)

    cv2.imshow('applyMaskToImage', result_image)

    return result_image