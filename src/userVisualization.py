import numpy as np
import cv2

from src.constants import OBJECT_COLORS

def applyContoursToImage(image, contours, human_binary_frame):
    # Convert the input image to a copy
    result_image = image.copy()

    human_contours, _ = cv2.findContours(human_binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and draw non-moving objects with different colors
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Exclude the human blob from being considered as a separate object
        if not is_contour_inside_human(contour, human_contours):

            # Additional conditions for excluding objects
            if area > 500 and 500 < perimeter < 2000:  # Adjust the conditions as needed
                # Get the aspect ratio of the bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h != 0 else 0.0

                # Additional conditions to exclude long and narrow objects
                if aspect_ratio < 5.0:
                    # Assign different colors based on the aspect ratio
                    if aspect_ratio > 1.5:
                        color_for_contour = (255, 0, 0)  # Blue for relatively wide objects
                    else:
                        color_for_contour = (0, 255, 0)  # Green for other objects
                    cv2.drawContours(result_image, [contour], -1, color_for_contour)

    # Draw the edge of the human contour with a specific color
    if human_binary_frame.any():
        human_color = (255, 255, 255)  # White color for the human
        cv2.drawContours(result_image, human_contours, -1, human_color, thickness=1)  # Thin edge of human contour


    cv2.imshow('applyContoursToImage', result_image)

    return result_image


def is_contour_inside_human(contour, human_contours):
    for human_contour in human_contours:
        # Check if any point of the contour is inside the human contour
        result = any(
            [cv2.pointPolygonTest(human_contour, (int(point[0][0]), int(point[0][1])), False) > 0 for point in contour]
        )
        if result:
            return True
    return False













