import cv2
import argparse
import numpy as np
# from tensorflow.keras.models import load_model

from visualizations import visualize_predictions

# Load the model
# model_path = '../neural_networks/human_model.h5'
# loaded_model = load_model(model_path)

# Create the background subtractor with selective updating
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=15,        # The number of last frames that affect the background model.
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


def handleBinaryMask_v2(binaryMaskRaw, dilation_iterations=1, min_blob_area=50):
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

    contours, _ = cv2.findContours(blobs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('Amount of contours: ', contours.__len__())

    cv2.imshow('blobs_mask', blobs_mask)
    return blobs_mask


def getBinaryMask(gray_frame_filtered):
    _, binary_mask = cv2.threshold(gray_frame_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('binary_mask', binary_mask)
    return binary_mask


def handleGrayscaleFiltering(gray_frame):
    # Apply bilateral and Gaussian filtering
    gray_frame_filtered = cv2.bilateralFilter(gray_frame, d=7, sigmaColor=45, sigmaSpace=45)
    return gray_frame_filtered


def handleEdgeDetection(gray_frame_filtered, lower_threshold=69, upper_threshold=70):
    # Apply Canny edge detection
    edges = cv2.Canny(gray_frame_filtered, lower_threshold, upper_threshold)
    # You can further process the 'edges' image if needed
    gray_frame_filtered = cv2.bitwise_and(gray_frame_filtered, gray_frame_filtered, mask=edges)
    cv2.imshow('gray_frame_edges', gray_frame_filtered)
    return gray_frame_filtered


def extractHumanObject(binary_mask, learningRate):
    fg_mask = bg_subtractor.apply(binary_mask, learningRate=learningRate)
    # Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    # Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('fg_mask', fg_mask)
    return fg_mask


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

        # Human operations
        human_binary_frame = extractHumanObject(gray_frame.copy(), learningRate=0.5)

        # Apply grayscale filtering operations
        gray_frame_filtered = handleGrayscaleFiltering(gray_frame)

        gray_frame_edges = handleEdgeDetection(gray_frame_filtered)

        # Convert grayscale to binary mask
        binary_mask_raw = getBinaryMask(gray_frame_edges)

        # Overlay binary_mask_raw with humanObject and exclude human from binary_mask_raw
        binary_mask_with_deleted_movement = cv2.bitwise_and(binary_mask_raw, cv2.bitwise_not(human_binary_frame))

        cv2.imshow('binary_mask_with_deleted_movement', binary_mask_with_deleted_movement)

        binary_frame_for_objects = handleBinaryMask_v2(binary_mask_with_deleted_movement)

        masked_frame = applyMaskToImage(frame, binary_frame_for_objects)

        cv2.imshow('maskedFrame', masked_frame)

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
