import argparse
import cv2
import numpy as np

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

def process_frame(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fgmask = fgbg.apply(gray_frame)

    # Apply GaussianBlur to reduce noise and help edge detection
    filtered_image = cv2.GaussianBlur(fgmask, (3, 3), 0)

    # Apply binarization using OTSU's method to calculate the correct threshold value
    _, binary_frame = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define a kernel for the closing operation
    kernel_size_close = 3
    kernel_close = np.ones((kernel_size_close, kernel_size_close), np.uint8)
    closed_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel_close)

    return closed_frame


def connectComponentsStats(binary_frame, original_frame):
    # Perform connected components labeling
    num_labels, labeled_frame, stats, centroids = cv2.connectedComponentsWithStats(binary_frame, connectivity=8)

    # Extract information about each connected component
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x, y, width, height = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]

        # Draw a bounding box around the component if its area is larger than a threshold
        if area > 100:  # You can adjust the area threshold as needed
            cv2.rectangle(original_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return original_frame


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break

        binary_frame = process_frame(frame)
        frame = connectComponentsStats(binary_frame, frame)

        # Display the processed frame with OpenCV (not working in Jupyter)
        cv2.imshow('Processed Frame', binary_frame)

        # Stop playing when 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="./intrusion.avi", help="path to video")
    args = parser.parse_args()
    play_video(args.video_path)
