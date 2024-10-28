import numpy as np
import matplotlib.pyplot as plt

# Function to apply a Sobel Filter
def apply_sobel_filter(image):
    img_array = np.array(image, dtype=float)

    # Defining the Sobel Filters/Kernels - Edge Detection
    sobel_x = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

    sobel_y = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

    image_height = img_array.shape[1]
    image_width = img_array.shape[2]

    # Initializing the Gradients as Array of zeroes
    gradient_x = np.zeros_like(img_array)
    gradient_y = np.zeros_like(img_array)

    for num in range(img_array.shape[0]):
        for x in range(1, image_height-1):
            for y in range(1, image_width-1):
                patch = img_array[num][x-1 : x+2, y-1 : y+2]
                gradient_x[num][x, y] = np.sum(patch * sobel_x)
                gradient_y[num][x, y] = np.sum(patch * sobel_y)

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    gradient_x = (gradient_x - np.min(gradient_x)) / (np.max(gradient_x) - np.min(gradient_x))
    gradient_y = (gradient_y - np.min(gradient_y)) / (np.max(gradient_y) - np.min(gradient_y))
    gradient_magnitude = (gradient_magnitude - np.min(gradient_magnitude)) / (np.max(gradient_magnitude) - np.min(gradient_magnitude))

    return gradient_x, gradient_y, gradient_magnitude


# Function to display graphs/plot the results
def display_results(image, gradient_x, gradient_y, gradient_magnitude):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')


    axes[0, 1].imshow(gradient_x, cmap='gray')
    axes[0, 1].set_title('Horizontal Edges (Sobel X)')
    axes[0, 1].axis('off')


    axes[1, 0].imshow(gradient_y, cmap='gray')
    axes[1, 0].set_title('Vertical Edges (Sobel Y)')
    axes[1, 0].axis('off')


    axes[1, 1].imshow(gradient_magnitude, cmap='gray')
    axes[1, 1].set_title('Edge Magnitude')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()