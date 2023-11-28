import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(predictions, window_size=10):
    # Create an array of frame indices
    frame_indices = np.arange(1, len(predictions) + 1)

    # Round predictions to 0 or 1 based on the threshold
    rounded_predictions = np.round(predictions)
    mean_value = np.mean(rounded_predictions)
    print('Mean classification value: ', mean_value)

    # Create a 2D scatter plot with frame indices on the x-axis and prediction values on the y-axis
    plt.scatter(frame_indices, predictions, c=predictions, cmap='RdYlGn', marker='o', edgecolors='k', linewidths=0.5)

    # Calculate the running average
    running_avg = np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')

    # Plot the running average as a line
    plt.plot(frame_indices[window_size-1:], running_avg, color='blue', label=f'Running Average (window={window_size})')

    # Add colorbar for reference
    cbar = plt.colorbar()
    cbar.set_label('Prediction Probability')

    # Customize the plot
    plt.xlabel('Frame Index')
    plt.ylabel('Prediction Probability')
    plt.title('Human Recognition Results with Running Average')
    plt.legend()

    # Show the plot
    plt.show()