import json
import datetime
import os

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


def writeOutputFile(data_structure, folder_path='../output_files'):
    '''
    Currently dumps data just as a JSON object.
    '''

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"The folder {folder_path} has been created.")

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{folder_path}/output_{formatted_datetime}.txt"

    # Open the file in write mode
    with open(filename, 'w') as file:
        # Use json.dump to write the dictionary to the file
        json.dump(data_structure, file)

    print(f"Dictionary has been written to {filename}")