import my_canny as my
import cv2

image = cv2.imread("images/lena.png")

# iterate over the thresholds
# low_threshold is 1/2 of high_threshold
for i in range(1, 256):
    canny_image = my.canny(image, i / 2, i)
    cv2.imwrite("images/lena/2x/{:03d}.png".format(i), canny_image)

# iterate over the thresholds
# low_threshold is 1/3 of high_threshold
for i in range(1, 256):
    canny_image = my.canny(image, i / 3, i)
    cv2.imwrite("images/lena/3x/{:03d}.png".format(i), canny_image)
