import json
import datetime
import os
import pandas as pd
import openpyxl
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

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


def writeOutputFileEXCEL(data_structure, folder_path='../output_files'):

    checkFolderExistence(folder_path)


    filename = formatFileName(folder_path, ".pdf")


    pdf = SimpleDocTemplate(filename, pagesize=letter)
    elements = []


    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])


    table_data = [['Frame Index', 'Number of Detected Objects', 'Object Identifier', 'Area', 'Perimeter', 'Classification']]

    for frame in data_structure:

        num_objects = frame['number_of_detected_objects']
        if num_objects > 0:
            for obj in frame['frame_data']:
                row = [frame['frame_index'], num_objects, obj['identifier'], obj['area'], obj['perimeter'], obj['classification']]
                table_data.append(row)

            style.add('SPAN', (0, len(table_data)-num_objects), (0, len(table_data)-1))
            style.add('SPAN', (1, len(table_data)-num_objects), (1, len(table_data)-1))


    t = Table(table_data)
    t.setStyle(style)

    elements.append(t)


    pdf.build(elements)

    print(f"PDF file has been created: {filename}")