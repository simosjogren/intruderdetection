import cv2
import argparse
import numpy as np
# from tensorflow.keras.models import load_model

from visualizations import visualize_predictions

# Load the model
model_path = '../neural_networks/human_model.h5'
# loaded_model = load_model(model_path)

# Create the background subtractor with selective updating
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=50,        # The number of last frames that affect the background model.
    varThreshold=-1,    # Mahalanobis distance threshold.
    detectShadows=False   # If True, the model will detect shadows and mark them as 127.
)


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


def performLabeling(value, threshold=0.5):
    if value >= threshold:
        # Human found
        return True
    else:
        # No human found
        return False
    

def handleBinaryMask(binary_mask):
    # Display the results
    cv2.imshow('binary_mask', binary_mask)

    # Perform opening operation to remove noise
    kernel = np.ones((2, 2), np.uint8)
    opening_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Assuming 'binary_mask' is your binary mask
    _, labeled_image = cv2.connectedComponents(opening_mask)
    blob_areas = [np.sum(labeled_image == label) for label in range(1, np.max(labeled_image) + 1)]
    # Define a threshold for blob density
    density_threshold = 10  # Iterated to this value
    # Identify regions of interest based on blob density
    roi_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for label, area in enumerate(blob_areas, start=1):
        if area > density_threshold:
            roi_mask[labeled_image == label] = 255

    # Y-dilation to extend the ROI vertically
    kernel_y_size = 5
    kernel_y = np.ones((kernel_y_size, 1), np.uint8)  # Adjust the kernel size based on your needs
    dilated_roi_y = cv2.dilate(roi_mask, kernel_y, iterations=1)

    # Closing operation for refinement
    kernel = np.ones((5, 5), np.uint8)
    closing_roi = cv2.morphologyEx(dilated_roi_y, cv2.MORPH_CLOSE, kernel)

    return closing_roi


def getBinaryMask(gray_frame_filtered, learningRate=0.15):
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(gray_frame_filtered, learningRate=learningRate)
    _, binary_mask = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask


def handleGrayscaleFiltering(gray_frame):
    # Apply bilateral and Gaussian filtering
    gray_frame_filtered = cv2.bilateralFilter(gray_frame, d=7, sigmaColor=45, sigmaSpace=60)
    return gray_frame_filtered

def applyMaskToImage(image, binary_mask, color_for_masked_region=[0, 0, 255]):
    # Convert the binary mask to a 3-channel image
    binary_mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Ensure that color_for_masked_region is a NumPy array with dtype=np.uint8
    color_for_masked_region = np.array(color_for_masked_region, dtype=np.uint8)

    # Use the binary mask to create a mask for the colored region
    masked_region = cv2.bitwise_and(binary_mask_color, color_for_masked_region)

    # Combine the masked region and the original image using bitwise_or
    result_image = cv2.bitwise_or(image, masked_region)

    return result_image



def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
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

        # Apply grayscale filtering operations
        gray_frame_filtered = handleGrayscaleFiltering(gray_frame)

        # Convert grayscale to binary mask
        binaryMaskRaw = getBinaryMask(gray_frame_filtered, learningRate=0.15)

        # Perform operations to the binary mask
        binaryFixed = handleBinaryMask(binaryMaskRaw)

        maskedFrame = applyMaskToImage(frame, binaryFixed)

        cv2.imshow('maskedFrame', maskedFrame)

        # Stop playing when 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break
        print(index)
        index += 1

    cap.release()
    cv2.destroyAllWindows()

    # You can add your own visualization code here based on prediction_results
    if (prediction_results is not None) and (len(prediction_results) > 0):
        visualize_predictions(prediction_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="../data/intrusion.avi", help="path to video")
    args = parser.parse_args()
    play_video(args.video_path)
