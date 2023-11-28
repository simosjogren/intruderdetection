import cv2
import numpy as np

'''
Useful functions for just-in-case situations.
'''


def rescaleImageForNN(image, target_size=(256, 256)):
    # Get the original dimensions
    height, width = image.shape[:2]

    # Calculate the scaling factors
    scale_x = target_size[0] / width
    scale_y = target_size[1] / height

    # Choose the minimum scaling factor to avoid distortion
    scale = min(scale_x, scale_y)

    # Resize the image with maintaining the aspect ratio
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)

    # Create a blank canvas of the target size
    canvas = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    # Calculate the position to paste the resized image
    x_offset = (canvas.shape[1] - resized_image.shape[1]) // 2
    y_offset = (canvas.shape[0] - resized_image.shape[0]) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1]] = resized_image
    return canvas


def extractROI(frame, binary_mask, padding=25):
    # Dilate and erode the binary mask to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)

    # Find contours on the processed mask
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # If no contours were found, return None
    if len(contours) == 0:
        return None

    # Filter contours based on area
    min_contour_area = 100  # Adjust the threshold as needed
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # If no filtered contours were found, return None
    if len(filtered_contours) == 0:
        return None

    # Combine all filtered contours into one
    combined_contour = np.concatenate(filtered_contours)

    # Get bounding rectangle around the combined contour with padding
    x, y, w, h = cv2.boundingRect(combined_contour)
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # Extract the region of interest (ROI) from the original image with padding
    roi = frame[max(0, y):min(frame.shape[0], y+h), max(0, x):min(frame.shape[1], x+w)]

    return roi