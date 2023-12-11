import cv2
import argparse
import numpy as np


# Local modules import
from src.outputFileMaker import makeFrameDict, writeOutputFileEXCEL, classify_objects
from src.humanSeparation import extractHumanObject
from src.objectSeparation import handleEdgeDetection, getBinaryMaskForObject, formBlobsAndContours, separateHumanFromObjectFrame
from src.userVisualization import applyContoursToImage
from src.preProcessingMethods import handleGrayscaleFiltering


# Create the background subtractor with selective updating
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=30,        # The number of last frames that affect the background model.
    varThreshold=15,    # Mahalanobis distance threshold.
    detectShadows=False   # If True, the model will detect shadows and mark them as 127.
)


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    index = 1
    output_file = []

    alpha = 0.9999  # Adaptation rate for blind background updating
    background_model = None
    previous_frame_data = None  # Initialize previous_frame_data

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if background_model is None:
            # Initialize the background model with the first frame
            background_model = gray_frame.astype(np.float32)

        # Apply basic grayscale filtering operations to the object recognition frame
        gray_frame_filtered = handleGrayscaleFiltering(gray_frame)

        # Lets separate the human out at this spot
        human_binary_frame, background_model = extractHumanObject(gray_frame, alpha, background_model)

        # Edge separation
        gray_frame_edges = handleEdgeDetection(gray_frame_filtered)

        # Convert grayscale -> binary mask
        binary_mask_raw = getBinaryMaskForObject(gray_frame_edges)

        # Overlay binary_mask_raw with separated human and exclude human from binary_mask_raw
        binary_mask_with_deleted_movement = separateHumanFromObjectFrame(binary_mask_raw, human_binary_frame)

        # Find and separate the contours (gives only raw format of contours, needs filtering.)
        object_contours, _ = formBlobsAndContours(binary_mask_with_deleted_movement)

        # Export the given data to dict format as a JSON like object for future file export.
        frameData = makeFrameDict(object_contours, human_binary_frame, index)
        output_file.append(frameData)

        # Pass current_frame_data and previous_frame_data to classifyBlob
        current_frame_data = {
            'frame_index': index,
            'number_of_detected_objects': len(object_contours),
            'frame_data': frameData['frame_data']
        }
        classify_objects(current_frame_data, previous_frame_data)

        previous_frame_data = current_frame_data


        # Represent the objects to the user
        applyContoursToImage(frame, object_contours, human_binary_frame)

        # Stop playing when 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break
        print(index)
        index += 1

    cap.release()
    cv2.destroyAllWindows()

    # Export the data to EXCEL file as good format.
    writeOutputFileEXCEL(output_file)