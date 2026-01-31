import torch
import matplotlib.pyplot as plt


def plot_results(model, distances, times):
    """
    Plots the actual data points and the model's prdicted line for a given dataset.

    Args:
    model: The trained machine learning model to use for predictions.
    distances: The input data points(features) for the model
    times: The target data points(labels) for the plot.
    """

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations for efficient inference
    with torch.no_grad():
        # Make predictions using the trained model
        predicted_times = model(distances)

    # Create  a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the actual data points
    plt.plot(distances.numpy(), times.numpy(), color='orange',
             marker='o', linestyle='None', label='Actual Delivery Times')

    # Plot the predicted line for the mpodel
    plt.plot(distances.numpy(), predicted_times.numpy(),
             color='green', marker='None', label='Predicted Line')

    # \set the title for the plot
    plt.title('Actual vs Predicted Delivery Times')
    # Set the x-axis label
    plt.xlabel('Distance (miles)')
    # Set the y-axis label
    plt.ylabel("Time(Minutes)")
    # Display the legend
    plt.legend()
    # Add a grid to the plot
    plt.grid(True)
    # Show the plot
    plt.show()


def plot_nonlinear_comparison(model, new_distances, new_times):
    """
    Compare and plots the predictions of a model against new, non-linear data.

    Args:
    model: The trained model to be evaluated.
    new_distances: The new input data for generating predictions
    new_times: The actual target values for comparison
    """
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation forinference
    with torch.no_grad():
        # Generate predictions using the model
        predictions = model(new_distances)

    # Create a neew figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the actual data points
    plt.plot(new_distances.numpy(), new_times.numpy(), color='orange',
             marker='o', linestyle='None', label='Actual Data (Bikes and Cars)')

    # Plot the predictions from the model
    plt.plot(new_distances.numpy(), predictions.numpy(), color='green',
             marker='None', label='Linear Model Predictions')

    # Set the title of the plot
    plt.title('Linear Model vs. Non-Linearity Reality')
    # Set the label for the x-axis
    plt.xlabel('Distance (miles)')
    # Set the label for the y-axis
    plt.ylabel('Time (minutes)')
    # Add a new legemd to the plot
    plt.legend()
    # Add a grid to the plot for better reliability
    plt.grid(True)
    # Display the plot
    plt.show()
