import cv2
import os
import numpy as np


'''
This solution is purely additional and hypotetical situation whetever the DoG & SIFT 
combination could be used.

'''


def getFileNames(path):
    file_names = []
    for filename in os.listdir(path):
        # You might want to skip directories if only files are needed
        if os.path.isfile(os.path.join(path, filename)):
            file = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            file_names.append([file])
    return file_names


def trainObjectRecognition(path='./query_frames'):
    '''
    This is based on DoG & SIFT
    '''
    query_images = getFileNames(path)
    sift = cv2.SIFT_create()

    # Lets count the keypoints
    for n in range(len(query_images)):
        kp_query = sift.detect(query_images[n][0])  # zero index is the stored frame
        query_images[n].append(kp_query)

    # Lets compute the descriptors & update the keypoints accordingly
    for n in range(len(query_images)):
        kp_query, des_query = sift.compute(query_images[n][0], query_images[n][1])
        query_images[n][1] = kp_query   # Updating the keypoints
        query_images[n].append(des_query)   # Desc as the third index

    # Initialize FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)   # Higher values gives better precision, but also takes more time
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    return query_images, flann, sift


def extractObjects(gray_frame, queryData, flann, sift, lowe_threshold=0.7):
    # Detect and compute keypoints and descriptors in gray_frame
    kp_frame = sift.detect(gray_frame)
    kp_frame, des_frame = sift.compute(gray_frame, kp_frame)

    for n in range(len(queryData)):
        # Find matches with FLANN
        matches = flann.knnMatch(queryData[n][2], des_frame, k=2)
        queryData[n].append(matches)

    # Filtering matches using Lowe's ratio test
    for i in range(len(queryData)):
        good_matches = [m for m, n in queryData[i][3] if m.distance < lowe_threshold * n.distance]
        queryData[i][3] = good_matches

        # Find homography and map the contours if enough matches are found
        MIN_MATCH_COUNT = 10
        print('matches: ', len(good_matches))
        if len(good_matches) > MIN_MATCH_COUNT:
            print("Found in the picture!")
            # Get the keypoints from the good matches
            src_pts = np.float32([queryData[i][1][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography matrix
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Map the contours (or bounding box) of the query image to the scene image
            h, w = queryData[i][0].shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Draw the contours on the gray_frame
            gray_frame = cv2.polylines(gray_frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

            queryData[i].append(True)  # Indicate match status
            return gray_frame  # Return the frame with drawn contours
        else:
            queryData[i].append(False)
    
    return None  # Return None if no object is found




