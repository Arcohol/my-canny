import my_canny as my
import cv2

image = cv2.imread("images/lena.png", 0)

def test_canny(image, start, stop):
    blur = my.gaussian_blur(image)
    gradient, direction = my.sobel(blur)
    non_max = my.non_max_suppresion_with_interpolation(gradient, direction)

    for i in range(start, stop):
        weak, strong = my.double_threshold(non_max, i / 2, i)
        canny_image = my.hysteresis(weak, strong)
        cv2.imwrite("images/lena/2x/{:03d}.png".format(i), canny_image)

        weak, strong = my.double_threshold(non_max, i / 3, i)
        canny_image = my.hysteresis(weak, strong)
        cv2.imwrite("images/lena/3x/{:03d}.png".format(i), canny_image)


test_canny(image, 1, 256)
