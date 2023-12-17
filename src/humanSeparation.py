import cv2
import numpy as np


def morphological_operations(binary_frame):
    '''
    Performs the necessary morphological operations to the binary frame
    '''
    # Opening
    kernel_size_opening = (2,2)
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_opening)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel_opening, iterations=2)

    # Closing
    kernel_size_closing = (5, 5)
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_closing)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel_closing, iterations=1)

    return binary_frame


def extractHumanObject(grayscale_frame, alpha, background_model, binary_threshold=32):
    # Convert background_model to uint8
    background_model_uint8 = background_model.astype(np.uint8)

    # Compute absolute difference
    diff_mask = cv2.absdiff(grayscale_frame, background_model_uint8)

    # Simple alpha blending for selective background updating
    alpha_mask = cv2.threshold(diff_mask, binary_threshold, 255, cv2.THRESH_BINARY)[1]
    background_model_update = alpha_mask / 255.0 * (grayscale_frame - background_model_uint8)

    # Update background only in regions where there is no foreground
    background_model += alpha * background_model_update

    # Apply threshold to convert fg_mask into binary frame
    _, binary_frame = cv2.threshold(diff_mask, binary_threshold, 255, cv2.THRESH_BINARY)

    binary_frame = morphological_operations(binary_frame)

    # Show the afterMorphologicalOperations mask
    cv2.imshow('afterMorphologicalOperations', binary_frame)

    # Perform edge detection on the binary mask
    edges = cv2.Canny(binary_frame, 30, 100)

    # Find contours of the edges
    contours_edges, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return np.zeros_like(binary_frame), background_model

    # Find the convex hull of the outermost points
    epsilon = 0.0001 * cv2.arcLength(np.vstack(contours_edges), True)  # Adjust epsilon for more or less detail
    approx_hull = cv2.approxPolyDP(np.vstack(contours_edges), epsilon, True)
    convex_hull = cv2.convexHull(approx_hull)

    # Create a new mask for the convex hull
    convex_hull_mask = np.zeros_like(edges)

    # Draw the convex hull on the new mask
    cv2.drawContours(convex_hull_mask, [convex_hull], -1, (255), thickness=1)

    return convex_hull_mask, background_model
