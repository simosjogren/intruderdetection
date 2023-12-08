import datetime
import os
import pandas as pd
import cv2


def makeFrameDict(contours, human_blob, frame_idx):
    ## TODO: Human object handling
    print("Index: ", frame_idx)

    amount_of_objects = contours.__len__()
    frame_data = []
    for n in range(amount_of_objects):
        object_data = {}
        object_data["identifier"] = n
        object_data["area"] = int(cv2.contourArea(contours[n]))
        object_data["perimeter"] = int(cv2.arcLength(contours[n], True))
        object_data["other_blob_features"] = [] # TODO
        object_data["classification"] = "object"
        frame_data.append(object_data)

    return {
        'frame_index': frame_idx,
        'number_of_detected_objects': amount_of_objects,
        'frame_data': frame_data
    }


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