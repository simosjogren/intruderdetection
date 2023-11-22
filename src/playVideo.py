import cv2
import argparse
import numpy as np
from tensorflow.keras.models import load_model

from visualizations import visualize_predictions

# Load the model
model_path = '../neural_networks/human_model.h5'
loaded_model = load_model(model_path)


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


def extractROI(frame, binary_mask, padding=15):
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


def performLabeling(value, threshold=0.5):
    if value >= threshold:
        # Human found
        return True
    else:
        # No human found
        return False


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Create the background subtractor with selective updating
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=50,        # The number of last frames that affect the background model.
        varThreshold=-1,    # Mahalanobis distance threshold.
        detectShadows=False   # If True, the model will detect shadows and mark them as 127.
    )

    learningRate = 0.04   # For the .apply phase. -1 means automatic learning rate.

    index = 1
    prediction_results = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral and Gaussian filtering
        gray_frame_filtered1 = cv2.bilateralFilter(gray_frame, d=9, sigmaColor=45, sigmaSpace=15)
        gray_frame_filtered2 = cv2.GaussianBlur(gray_frame_filtered1, (5, 5), 0)

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(gray_frame_filtered2, learningRate=learningRate)

        _, binary_mask = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Define a kernel for the opening operation
        kernel_size_open = 3
        kernel_open = np.ones((kernel_size_open, kernel_size_open), np.uint8)
        opened_frame = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)

        # Define a kernel for the closing operation
        kernel_size_close = 9
        kernel_close = np.ones((kernel_size_close, kernel_size_close), np.uint8)
        closed_frame = cv2.morphologyEx(opened_frame, cv2.MORPH_CLOSE, kernel_close)

        # Assuming you have a frame and a binary_mask
        extracted_area = extractROI(gray_frame, binary_mask)

        # If the extracted_area is not None, you can use it
        if extracted_area is not None:
            resized_image = rescaleImageForNN(extracted_area)
            # Normalize pixel values to be between 0 and 1
            normalized_image = resized_image / 255.0
            # Expand dimensions to create a batch size of 1
            input_image = np.expand_dims(normalized_image, axis=0)
            predictions = loaded_model.predict(input_image)
            prediction_results.append(predictions[0][0])
            if performLabeling(predictions[0][0]):
                print("Human found")
            cv2.imshow('Extracted Area', resized_image)
            # cv2.waitKey(0)

        # Display the results
        cv2.imshow('Difference', closed_frame)

        # Stop playing when 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break
        print(index)
        index += 1

    cap.release()
    cv2.destroyAllWindows()

    # You can add your own visualization code here based on prediction_results
    visualize_predictions(prediction_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="../data/intrusion.avi", help="path to video")
    args = parser.parse_args()
    play_video(args.video_path)
