import cv2
import argparse
import numpy as np

# Load the background model
# background_model = np.load('../data/background_model.npy')


def drawContours(frame, closed_frame):
        # Find contours in the closed frame
        contours, _ = cv2.findContours(closed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out contours near the borders
        frame_height, frame_width = frame.shape[:2]
        border_margin = 10  # Adjust this margin based on your needs

        filtered_contours = [contour for contour in contours if
                             all(border_margin < point[0][0] < frame_width - border_margin and
                                 border_margin < point[0][1] < frame_height - border_margin
                                 for point in contour)]

        # Draw the filtered contours on a copy of the original frame
        frame_with_contours = frame.copy()
        cv2.drawContours(frame_with_contours, filtered_contours, -1, (0, 255, 0), 2)
        return frame_with_contours


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    '''
    Increased history causes the model to take longer to adapt to a changing background.
    Decreased history causes the model to adapt to changes in the background faster.
    Increased varThreshold causes the model to be more sensitive to changes in the background.
    Decreased varThreshold causes the model to be less sensitive to changes in the background.
    '''

    # Create the background subtractor with selective updating
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100,    # If over 100, the model will take enough time to recover from the overlighting situation.
        detectShadows=False
    )

    learningRate = 0.05   # For the .apply phase. -1 means automatic learning rate.

    index = 1
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        '''
        Larger d: Wider pixel neighborhood for filtering.
        Larger sigmaColor: More distant colors are considered, leading to more color smoothing.
        Larger sigmaSpace: Pixels farther away from the central pixel influence the smoothing more, 
        leading to more spatial smoothing.
        '''
        filtered_image = cv2.bilateralFilter(gray_frame, d=7, sigmaColor=15, sigmaSpace=70)

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(filtered_image, learningRate=learningRate)

        _, binary_mask = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Define a kernel for the opening operation
        kernel_size_open = 3
        kernel_open = np.ones((kernel_size_open, kernel_size_open), np.uint8)
        opened_frame = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)

        # Define a kernel for the closing operation
        kernel_size_close = 5
        kernel_close = np.ones((kernel_size_close, kernel_size_close), np.uint8)
        closed_frame = cv2.morphologyEx(opened_frame, cv2.MORPH_CLOSE, kernel_close)

        # frame_with_contours = drawContours(frame, closed_frame)

        # Display the results
        cv2.imshow('Difference', closed_frame)
        # cv2.imshow('Frame with Contours', frame_with_contours)


        # Stop playing when 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break
        print(index)
        index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="../data/intrusion.avi", help="path to video")
    args = parser.parse_args()
    play_video(args.video_path)
