import cv2

'''
These methods are applied before giving the frame to object recognition 'loop'.
'''

def handleGrayscaleFiltering(gray_frame):
    # Apply bilateral and Gaussian filtering
    gray_frame_filtered = cv2.bilateralFilter(gray_frame, d=7, sigmaColor=45, sigmaSpace=45)
    return gray_frame_filtered