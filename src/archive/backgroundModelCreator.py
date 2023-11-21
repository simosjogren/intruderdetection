import cv2
import numpy as np

def create_background_model_from_frame(video_path, num_frames=25, start_frame=0):
    cap = cv2.VideoCapture(video_path)

    # Skip to the specified starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read the frames to initialize the background model
    background_model = None
    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        if background_model is None:
            background_model = frame.astype(np.float32)
        else:
            background_model += frame
        frame_count += 1

    # Average the frames to create the background model
    background_model /= frame_count

    cap.release()

    return background_model.astype(np.uint8)


def save_background_model(background_model, filename='../data/background_model.npy'):
    np.save(filename, background_model)

def load_background_model(filename='background_model.npy'):
    return np.load(filename)


video_path = '../data/intrusion.avi'
print('Loading video from {}'.format(video_path))

# We know that the screen is empty at 256.
background_model = create_background_model_from_frame(video_path, start_frame=256)

# Save the background model
filename = '../data/background_model.npy'
print('Saving background model to', filename)
save_background_model(background_model, filename=filename)