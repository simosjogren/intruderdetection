import cv2
import numpy as np


def morphological_operations(binary_frame, opening_kernel_size=(3, 3), closing_kernel_size=(9, 9), iterations=1):
    # Opening
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel_size)
    result = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel_opening, iterations=iterations)

    # Closing
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, closing_kernel_size)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_closing, iterations=iterations)

    return result


def extractHumanObject(grayscale_frame, alpha, background_model, binary_threshold=30):
    # Convert background_model to uint8
    background_model_uint8 = background_model.astype(np.uint8)

    # Compute absolute difference
    fg_mask = cv2.absdiff(grayscale_frame, background_model_uint8)

    # Simple alpha blending for blind background updating
    background_model = (1 - alpha) * grayscale_frame + (alpha * background_model)

    # Apply threshold to convert fg_mask into binary frame
    _, binary_frame = cv2.threshold(fg_mask, binary_threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological operations
    fg_mask = morphological_operations(binary_frame)

    # Find contours of the separated blobs
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on an empty mask
    mask = np.zeros_like(fg_mask)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Show the filled mask
    cv2.imshow('extractHumanObject', mask)

    return mask, background_model