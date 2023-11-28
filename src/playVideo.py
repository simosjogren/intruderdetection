import cv2
import argparse
import numpy as np


# Local modules import
from src.outputFileMaker import makeFrameDict, writeOutputFileEXCEL
from src.humanSeparation import extractHumanObject
from src.objectSeparation import handleEdgeDetection, getBinaryMask, formBlobsAndContours, separateHumanFromObjectFrame
from src.userVisualization import applyMaskToImage
from src.preProcessingMethods import handleGrayscaleFiltering


# Create the background subtractor with selective updating
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=15,        # The number of last frames that affect the background model.
    varThreshold=-1,    # Mahalanobis distance threshold.
    detectShadows=False   # If True, the model will detect shadows and mark them as 127.
)


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    index = 1
    output_file = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply basic grayscale filtering operations to the object recognition frame
        gray_frame_filtered = handleGrayscaleFiltering(gray_frame)

        # Lets separate the human out at this spot
        human_binary_frame = extractHumanObject(gray_frame_filtered.copy(), bg_subtractor, learningRate=0.5)

        # Edge separation
        gray_frame_edges = handleEdgeDetection(gray_frame_filtered)

        # Convert grayscale -> binary mask
        binary_mask_raw = getBinaryMask(gray_frame_edges)

        # Overlay binary_mask_raw with separated human and exclude human from binary_mask_raw
        binary_mask_with_deleted_movement = separateHumanFromObjectFrame(binary_mask_raw, human_binary_frame)

        # Find and separate the contours (gives only raw format of contours, needs filtering.)
        contours, binary_frame_for_objects = formBlobsAndContours(binary_mask_with_deleted_movement)

        # TODO: Perform contour filtering: get small contours off, try to locate the most relevant ones only
        # TODO: Count atleast area & perimeter for every recognized object.

        # Apply the blob/object recognition to the actual frame & represent it to user.
        outputImage = applyMaskToImage(frame, binary_frame_for_objects)

        # Export the given data to dict format as a JSON like object for future file export.
        frameData = makeFrameDict(contours, human_binary_frame, index)
        output_file.append(frameData)

        # Stop playing when 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break
        print(index)
        index += 1

    cap.release()
    cv2.destroyAllWindows()

    # Export the data to EXCEL file as good format.
    writeOutputFileEXCEL(output_file)