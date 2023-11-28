import cv2

def extractHumanObject(binary_mask, bg_subtractor, learningRate):
    fg_mask = bg_subtractor.apply(binary_mask, learningRate=learningRate)
    # Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    # Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('fg_mask', fg_mask)
    return fg_mask