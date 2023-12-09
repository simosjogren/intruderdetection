import numpy as np
import cv2

from src.constants import OBJECT_COLORS

def applyContoursToImage(image, contours, human_binary_frame):
    # Convert the input image to a copy
    result_image = image.copy()

    human_contours, _ = cv2.findContours(human_binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the human contour with a specific color
    if human_contours:
        human_color = OBJECT_COLORS[0]  # You can choose a specific color for the human
        cv2.drawContours(result_image, human_contours, -1, human_color)

    # Iterate through contours and draw them with unique colors
    for i, contour in enumerate(contours):
        color_for_contour = OBJECT_COLORS[(i + 1) % len(OBJECT_COLORS)]  # Start from the second color
        cv2.drawContours(result_image, [contour], -1, color_for_contour)

    cv2.imshow('applyContoursToImage', result_image)

    return result_image

