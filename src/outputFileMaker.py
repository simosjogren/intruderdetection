import datetime
import os
import pandas as pd
import numpy as np
import cv2

def makeFrameDict(contours, human_blob, frame_idx):
    print("Index: ", frame_idx)

    amount_of_objects = len(contours)  # Use len() directly for simplicity
    frame_data = []

    for n, contour in enumerate(contours):
        object_data = {}
        object_data["identifier"] = n
        object_data["area"] = int(cv2.contourArea(contour))
        object_data["perimeter"] = int(cv2.arcLength(contour, True))

        # Compute the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Compute additional features
        aspect_ratio = float(w) / h if h != 0 else 0.0
        circularity = 4 * np.pi * object_data["area"] / (object_data["perimeter"] ** 2) if object_data["perimeter"] != 0 else 0.0

        # Solidity is the ratio of contour area to its convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = object_data["area"] / hull_area if hull_area != 0 else 0.0

        # Add features to the object_data dictionary
        object_data["aspect_ratio"] = aspect_ratio
        object_data["circularity"] = circularity
        object_data["solidity"] = solidity

        # Perform classification based on features
        classification = classifyBlob(object_data)  # Implement your classification logic

        object_data["classification"] = classification
        frame_data.append(object_data)

    return {
        'frame_index': frame_idx,
        'number_of_detected_objects': amount_of_objects,
        'frame_data': frame_data
    }


def classifyBlob(blob_data):
    # Example feature extraction
    area = blob_data["area"]
    perimeter = blob_data["perimeter"]
    aspect_ratio = blob_data["aspect_ratio"]  # Add this feature to the makeFrameDict function

    # Example classification logic
    if area > 500 and perimeter > 100:
        return "person"
    elif area > 200 and aspect_ratio > 0.5:
        return "person"
    else:
        return "other"


def checkFolderExistence(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"The folder {folder_path} has been created.")
        return True
    return False


def formatFileName(folder_path, fileNameEnding):
    '''
    Include dot to the fileNameEnding, for example
    ".txt" or ".xlsx"
    '''
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{folder_path}/output_{formatted_datetime}{fileNameEnding}"
    return filename


def writeOutputFileEXCEL(data_structure, folder_path='./output_files'): #consider adding coloring to the columns here

    checkFolderExistence(folder_path)
    filename = formatFileName(folder_path, ".xlsx")

    formatted_rows = []

    current_row = 1
    for frame in data_structure:
        frame_idx = frame['frame_index']
        number_of_objects = frame['number_of_detected_objects']

        for obj in frame['frame_data']:
            row = {
                'frame_index': frame_idx,
                'number_of_detected_objects': number_of_objects,
                'object_identifier': obj['identifier'],
                'area': obj['area'],
                'perimeter': obj['perimeter'],
                'classification': obj['classification']
            }
            formatted_rows.append(row)

        if number_of_objects > 0:
            current_row += number_of_objects

    df = pd.DataFrame(formatted_rows)

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        worksheet = writer.sheets['Sheet1']
        current_row = 2
        for frame in data_structure:
            number_of_objects = frame['number_of_detected_objects']
            if number_of_objects > 0:
                end_row = current_row + number_of_objects - 1
                worksheet.merge_range(f'A{current_row}:A{end_row}', frame['frame_index'])
                worksheet.merge_range(f'B{current_row}:B{end_row}', number_of_objects)
                current_row += number_of_objects

    print(f"Excel file has been created: {filename}")