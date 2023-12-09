import cv2
import numpy as np


def morphological_operations(binary_frame, opening_kernel_size=(3, 3), closing_kernel_size=(12, 12), iterations=1):
    # Opening
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel_size)
    result = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel_opening, iterations=iterations)

    # Closing
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, closing_kernel_size)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_closing, iterations=iterations)

    # Perform vertical closing
    kernel_vertical_closing = np.ones((1,3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_vertical_closing)

    return result


def extractHumanObject(grayscale_frame, alpha, background_model, binary_threshold=32):
    # Convert background_model to uint8
    background_model_uint8 = background_model.astype(np.uint8)

    # Compute absolute difference
    fg_mask = cv2.absdiff(grayscale_frame, background_model_uint8)

    # Simple alpha blending for selective background updating
    alpha_mask = cv2.threshold(fg_mask, binary_threshold, 255, cv2.THRESH_BINARY)[1]
    background_model_update = alpha_mask / 255.0 * (grayscale_frame - background_model_uint8)

    # Update background only in regions where there is no foreground
    background_model += alpha * background_model_update

    # Apply threshold to convert fg_mask into binary frame
    _, binary_frame = cv2.threshold(fg_mask, binary_threshold, 255, cv2.THRESH_BINARY)

    # Opening
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    fg_mask = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel_opening, iterations=2)

    # Closing
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_closing, iterations=1)

    # Show the afterMorphologicalOperations mask
    cv2.imshow('afterMorphologicalOperations', fg_mask)

    # Perform edge detection on the binary mask
    edges = cv2.Canny(fg_mask, 30, 100)

    # Find contours of the edges
    contours_edges, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return np.zeros_like(fg_mask), background_model

    # Find the convex hull of the outermost points
    epsilon = 0.0001 * cv2.arcLength(np.vstack(contours_edges), True)  # Adjust epsilon for more or less detail
    approx_hull = cv2.approxPolyDP(np.vstack(contours_edges), epsilon, True)
    convex_hull = cv2.convexHull(approx_hull)

    # Create a new mask for the convex hull
    convex_hull_mask = np.zeros_like(edges)

    # Draw the convex hull on the new mask
    cv2.drawContours(convex_hull_mask, [convex_hull], -1, (255), thickness=1)

    # Show the convex hull mask
    cv2.imshow('convexHull', convex_hull_mask)

    return convex_hull_mask, background_model
