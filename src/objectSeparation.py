import cv2
import numpy as np

def handleEdgeDetection(gray_frame_filtered, lower_threshold=69, upper_threshold=70):
    '''
    Separates edges from the picture.
    '''
    # Apply Canny edge detection
    edges = cv2.Canny(gray_frame_filtered, lower_threshold, upper_threshold)
    # You can further process the 'edges' image if needed
    gray_frame_filtered = cv2.bitwise_and(gray_frame_filtered, gray_frame_filtered, mask=edges)
    cv2.imshow('handleEdgeDetection', gray_frame_filtered)
    return gray_frame_filtered


def getBinaryMaskForObject(gray_frame_filtered):
    '''
    Converts gray-scaled frame for binary mask
    '''
    _, binary_mask = cv2.threshold(gray_frame_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('getBinaryMask', binary_mask)
    return binary_mask


def separateHumanFromObjectFrame(binary_mask_raw, human_binary_frame):
    binary_mask_with_deleted_movement = cv2.bitwise_and(binary_mask_raw, cv2.bitwise_not(human_binary_frame))
    cv2.imshow('separateHumanFromObjectFrame', binary_mask_with_deleted_movement)
    return binary_mask_with_deleted_movement


def formBlobsAndContours(binaryMaskRaw, dilation_iterations=1, min_blob_area=50, non_moving_color=(0, 255, 0)):
    # Apply dilation to connect nearby edges
    dilated_edges = cv2.dilate(binaryMaskRaw, None, iterations=dilation_iterations)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the blobs
    blobs_mask = np.zeros_like(binaryMaskRaw)

    # Iterate through contours and draw filled blobs
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_blob_area:
            cv2.drawContours(blobs_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Highlight non-moving objects with a specific color
    non_moving_objects_mask = (binaryMaskRaw - blobs_mask) > 0
    blobs_mask_colored = cv2.cvtColor(blobs_mask, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels

    # Create a separate array for color changes
    color_changes = np.zeros_like(blobs_mask_colored)
    color_changes[non_moving_objects_mask] = non_moving_color

    # Combine the two arrays to get the final result
    result_mask = blobs_mask_colored + color_changes

    contours, _ = cv2.findContours(blobs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('Amount of contours: ', len(contours))

    cv2.imshow('formBlobsAndContours', result_mask)
    return contours, result_mask




