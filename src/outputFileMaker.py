import datetime
import os
import pandas as pd
import numpy as np
import cv2

def makeFrameDict(contours, human_contours, frame_idx):
    print("Index: ", frame_idx)

    amount_of_objects = len(contours)  # Use len() directly for simplicity
    frame_data = []
    human_found = 0     # This needs to be 0 if human didnt found, 1 if found.

    # Find the human contours. We capture the human contours only from the moving item
    if (len(human_contours) > 0):
        human_found = 1
        human_contour = max(human_contours, key=cv2.contourArea)   # Filters out the noise contours

        # Lets handle the moving human object separately
        object_data = {}
        object_data["identifier"] = 0
        object_data["area"] = int(cv2.contourArea(human_contour))
        object_data["perimeter"] = int(cv2.arcLength(human_contour, True))
        object_data["classification"] = 'human'

        # TODO: Implement the aspectratio, etc extras.
    
        frame_data.append(object_data)

    amount_of_objects += human_found

    # Find the object contours
    for n, contour in enumerate(contours):
        object_data = {}
        object_data["identifier"] = n + human_found     # human_found is incremental step
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

        object_data["classification"] = 'object'
        frame_data.append(object_data)

    return {
        'frame_index': frame_idx,
        'number_of_detected_objects': amount_of_objects,
        'frame_data': frame_data
    }


def classifyBlob(blob_data, frame_index):
    # TODO: FIX TO WORK LIKE WITH THE CURRENT IMPLEMENTATION
    area = blob_data["area"]
    perimeter = blob_data["perimeter"]
    aspect_ratio = blob_data["aspect_ratio"]

    # Example: Dynamic thresholds based on aspect ratio
    aspect_ratio_threshold = 0.6 if frame_index < 5 else 0.4

    # Example: Check if the blob characteristics are consistent with a human
    if aspect_ratio > aspect_ratio_threshold and area > 300 and perimeter > 80:
        return "person"
    else:
        return "other"


def classify_objects(current_frame_data, previous_frame_data):
    # TODO: FIX TO WORK LIKE WITH THE CURRENT IMPLEMENTATION
    frame_index = current_frame_data['frame_index']

    for obj in current_frame_data['frame_data']:
        # Extract features from the current object
        area = obj['area']
        perimeter = obj['perimeter']
        aspect_ratio = obj['aspect_ratio']  # Ensure that aspect_ratio is calculated in makeFrameDict

        # Extract features from the same object in the previous frame if available
        if previous_frame_data:
            previous_obj = find_matching_object(obj, previous_frame_data['frame_data'])
            if previous_obj:
                previous_area = previous_obj['area']
                previous_perimeter = previous_obj['perimeter']
                # Add more features as needed

        # Use features for classification logic (modify as needed)
        classification_result = classifyBlob(obj, frame_index)

        # Add the classification result to the object data
        obj['classification'] = classification_result

# Helper function to find the matching object in the previous frame
def find_matching_object(current_obj, previous_frame_data):
    for obj in previous_frame_data:
        if obj['identifier'] == current_obj['identifier']:
            return obj
    return None


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