import cv2
import numpy as np
from scipy.signal import convolve2d
import sys


def gaussian_blur(img):
    # 5x5 kernel
    gauss_mask = (
        np.array(
            [
                [2, 4, 5, 4, 2],
                [4, 9, 12, 9, 4],
                [5, 12, 15, 12, 5],
                [4, 9, 12, 9, 4],
                [2, 4, 5, 4, 2],
            ]
        )
        / 159
    )

    # Since the kernel is 2 more than the image, we pad the image with 2 using reflection
    padded = np.pad(img, 2, mode="reflect")

    out = convolve2d(padded, gauss_mask, mode="valid")

    return out


def sobel(img, l2gradient=True):
    # Sobel operator in x direction
    sobel_x = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]
    )

    # Sobel operator in y direction
    sobel_y = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ]
    )

    # Pad the image with 1
    padded = np.pad(img, 1, mode="reflect")

    # Convolve the image with the sobel operators
    dx = convolve2d(padded, sobel_x, mode="valid")
    dy = convolve2d(padded, sobel_y, mode="valid")

    # Compute the magnitude of the gradient
    if l2gradient:
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
    else:
        gradient_magnitude = np.abs(dx) + np.abs(dy)

    # Compute the direction of the gradient
    gradient_direction = np.arctan2(dy, dx)

    # Normalize the magnitude of the gradient to 255
    gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude) * 255

    return gradient_magnitude, gradient_direction


def non_max_suppresion(m, d):
    """
    m: gradient magnitude, d: gradient direction
    this function rounds the gradient direction to the nearest 45 degrees
    and checks if the pixel is a local maximum in the direction of the gradient
    """
    deg = np.rad2deg(d)
    deg[deg < 0] += 180
    deg = np.round(deg / 45) * 45

    # Create a new image of zeros with the same shape as the original image
    new_image = np.zeros_like(m)

    count = 0

    h, w = m.shape
    # Iterate over the image pixels
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            direction = deg[i, j]

            if direction == 0 or direction == 180:
                is_max = m[i, j] >= m[i, j - 1] and m[i, j] >= m[i, j + 1]
            elif direction == 45:
                is_max = m[i, j] >= m[i + 1, j + 1] and m[i, j] >= m[i - 1, j - 1]
            elif direction == 90:
                is_max = m[i, j] >= m[i - 1, j] and m[i, j] >= m[i + 1, j]
            elif direction == 135:
                is_max = m[i, j] >= m[i + 1, j - 1] and m[i, j] >= m[i - 1, j + 1]

            # Count the pixels that are being pruned
            count += is_max

            # Set the pixel to 0 if it is not a maximum
            new_image[i, j] = m[i, j] * is_max

    print("non-maximum suppression:", count)

    return new_image


def non_max_suppresion_with_interpolation(m, d):
    """
    m: gradient magnitude, d: gradient direction
    this function uses interpolation to check if the pixel is a local maximum
    in the direction of the gradient

    Note: this can only improve the results a little bit
    """
    deg = np.copy(d)
    deg[deg < 0] += np.pi

    new_image = np.zeros_like(m)

    count = 0

    h, w = m.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            direction = deg[i, j]

            # if direction equals 0
            if direction == 0:
                is_max = m[i, j] >= m[i, j - 1] and m[i, j] >= m[i, j + 1]

            # if direction belongs to (0, 45]
            elif 0 < direction <= np.pi / 4:
                r = direction
                x = (1 - np.tan(r)) * m[i, j + 1] + np.tan(r) * m[i + 1, j + 1]
                y = (1 - np.tan(r)) * m[i, j - 1] + np.tan(r) * m[i - 1, j - 1]
                is_max = m[i, j] >= x and m[i, j] >= y

            # if direction belongs to (45, 90)
            elif np.pi / 4 < direction < np.pi / 2:
                r = np.pi / 2 - direction
                x = (1 - np.tan(r)) * m[i + 1, j] + np.tan(r) * m[i + 1, j + 1]
                y = (1 - np.tan(r)) * m[i - 1, j] + np.tan(r) * m[i - 1, j - 1]
                is_max = m[i, j] >= x and m[i, j] >= y

            # if direction equals 90
            elif direction == np.pi / 2:
                is_max = m[i, j] >= m[i - 1, j] and m[i, j] >= m[i + 1, j]

            # if direction belongs to (90, 135)
            elif np.pi / 2 < direction < 3 * np.pi / 4:
                r = direction - np.pi / 2
                x = (1 - np.tan(r)) * m[i + 1, j] + np.tan(r) * m[i + 1, j - 1]
                y = (1 - np.tan(r)) * m[i - 1, j] + np.tan(r) * m[i - 1, j + 1]
                is_max = m[i, j] >= x and m[i, j] >= y

            # if direction belongs to [135, 180)
            elif 3 * np.pi / 4 <= direction < np.pi:
                r = np.pi - direction
                x = (1 - np.tan(r)) * m[i, j - 1] + np.tan(r) * m[i + 1, j - 1]
                y = (1 - np.tan(r)) * m[i, j + 1] + np.tan(r) * m[i - 1, j + 1]
                is_max = m[i, j] >= x and m[i, j] >= y

            # if direction equals 180
            elif direction == np.pi:
                is_max = m[i, j] >= m[i, j - 1] and m[i, j] >= m[i, j + 1]

            # Count the pixels that are being pruned
            count += is_max

            new_image[i, j] = m[i, j] * is_max

    print("non-maximum suppression with interpolation:", count)

    return new_image


def double_threshold(m, low_threshold, high_threshold):
    print("low_threshold:", low_threshold, "high_threshold:", high_threshold)

    weak_edges = np.zeros_like(m)
    strong_edges = np.zeros_like(m)

    # Get the weak edges
    weak_edges[np.logical_and(m >= low_threshold, m < high_threshold)] = 1

    # Get the strong edges
    strong_edges[m >= high_threshold] = 1

    return weak_edges, strong_edges


def hysteresis(weak, strong):
    while True:
        # Get the indices of the weak edges
        indices = np.argwhere(weak > 0)
        found = False
        # Iterate over the weak edges
        for i, j in indices:
            neighborhood = strong[i - 1 : i + 2, j - 1 : j + 2]
            if np.any(neighborhood > 0):
                found = True
                # Set the weak edge to a strong edge
                weak[i, j] = 0
                strong[i, j] = 1

        # If there are no new strong edges, we are done
        if not found:
            break

    strong[strong > 0] = 255

    return strong


def canny(image, low_threshold, high_threshold, interpolation=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    m, d = sobel(gaussian_blur(image))

    if interpolation:
        m = non_max_suppresion_with_interpolation(m, d)
    else:
        m = non_max_suppresion(m, d)

    weak, strong = double_threshold(m, low_threshold, high_threshold)
    edge = hysteresis(weak, strong)

    # return as uint8
    return edge.astype(np.uint8)


if __name__ == "__main__":
    # receive 3 arguments
    # 1. image path
    # 2. low threshold
    # 3. high threshold

    # read the image
    image = cv2.imread(sys.argv[1])
    low_threshold = int(sys.argv[2])
    high_threshold = int(sys.argv[3])

    # apply canny edge detection
    canny_image = canny(image, low_threshold, high_threshold)

    # save the image
    cv2.imwrite("out.png", canny_image)
