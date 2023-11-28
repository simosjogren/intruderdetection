import json
import datetime
import os
import pandas as pd

def makeFrameDict(contours, human_blob, frame_idx):
    ## TODO: Human object handling
    print("Index: ", frame_idx)
    print('Amount of contours: ', contours.__len__())
    amount_of_objects = contours.__len__()
    frame_data = []
    for n in range(amount_of_objects):
        object_data = {}
        object_data["identifier"] = n
        object_data["area"] = 100   # TODO
        object_data["perimeter"] = 100  # TODO
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


def writeOutputFileJSON(data_structure, folder_path='../output_files'):
    '''
    Currently dumps data just as a JSON object to a file.
    Preferred to use the EXCEL type, use only this if problems with Pandas.
    '''
    checkFolderExistence(folder_path)
    filename = formatFileName(folder_path, ".txt")

    # Open the file in write mode
    with open(filename, 'w') as file:
        # Use json.dump to write the dictionary to the file
        json.dump(data_structure, file)

    print(f"Dictionary has been written to {filename}")


def writeOutputFileEXCEL(data_structure, folder_path='../output_files'):
    '''
    Currently dumps data as EXCEL format to a file.
    '''
    checkFolderExistence(folder_path)
    filename = formatFileName(folder_path, ".xlsx")

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data_structure)

    # Write the DataFrame to an Excel file
    df.to_excel(filename, index=False)

    print(f"Excel file has been created: {filename}")