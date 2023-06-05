import cv2
import sys
import my_canny

if __name__ == "__main__":
    # 2 args, input image and output image
    sample = cv2.imread(sys.argv[1])

    edges = my_canny.canny(sample, 50, 100)

    cv2.imwrite("./edges.png", edges)

    element_x = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    element_y = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20))

    # Perform morphological operations
    img1 = cv2.dilate(edges, element_x, iterations=2)
    img2 = cv2.erode(img1, element_x, iterations=6)
    img3 = cv2.dilate(img2, element_x, iterations=4)
    img4 = cv2.erode(img3, element_y, iterations=2)
    img5 = cv2.dilate(img4, element_y, iterations=3)

    blur = cv2.medianBlur(img5, 15)

    # Find contours
    contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        r = w / h
        if 2 < r < 5:
            cv2.rectangle(sample, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(sys.argv[2], sample)
