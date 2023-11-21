import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()


# Custom loss function for binary classification and bounding box regression
def custom_loss(y_true, y_pred):
    # Binary cross-entropy loss for classification
    binary_loss = tf.keras.losses.binary_crossentropy(y_true[:, :1], y_pred[:, :1])
    # Mean squared error for bounding box coordinates
    regression_loss = tf.keras.losses.mean_squared_error(y_true[:, 1:], y_pred[:, 1:])
    return binary_loss + regression_loss


# Load the human recognition model
path = 'C://Users//simos//Desktop//VAIHTO//KOULU//Intruder detection git//intruderdetection//trained_model'
loaded_model = load_model(path, custom_objects={'custom_loss': custom_loss})


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


def normalize_images(images):
    return images.astype("float32") / 255.0


def evaluate_frame(frame, model):

    # Resize the frame to match the model's input size
    resized_frame = cv2.resize(frame, (256, 256))

    # Add channel dimension
    resized_frame = resized_frame[:, :, np.newaxis]

    # Normalize the frame
    normalized_frame = normalize_images(resized_frame)

    # Expand dimensions to create a batch with a single image
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Add an extra dimension for the batch

    # Make predictions
    predictions = model.predict(input_frame)

    # Interpret the predictions
    classification_prediction = predictions[0, 0]  # Assuming the first output is for classification
    regression_prediction = predictions[0, 1:]  # Assuming the second output is for bounding box coordinates

    return classification_prediction, regression_prediction


def draw_rectangle(image, coordinates):
    # Convert regression predictions to integers
    x, y, w, h = map(int, coordinates)

    # Draw the rectangle on the grayscale image
    color = 255  # White color in grayscale
    thickness = 2
    image_with_rectangle = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), color, thickness)

    return image_with_rectangle


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Evaluate the frame using the loaded model
        classification_prediction, regression_prediction = evaluate_frame(gray_frame, loaded_model)

        regression_prediction = regression_prediction * 255

        # Print the evaluation results
        print("Classification Prediction:", classification_prediction)
        print("Regression Prediction (Bounding Box Coordinates):", regression_prediction)

        # Draw the rectangle on the image
        frame_with_rectangle = draw_rectangle(gray_frame, regression_prediction)

        # Display the frame with OpenCV
        cv2.imshow('Frame', frame_with_rectangle)

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
