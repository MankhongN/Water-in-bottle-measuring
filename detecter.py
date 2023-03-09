import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
import imutils


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def pre_process_image(image, convert_type, lower=np.array([0, 0, 0]), upper=np.array([100, 100, 100])):
    img = image.copy()
    edge = None
    # if convert_type == 'color_pick':
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     img_color = cv2.inRange(img, lower, upper)
    #     blur = cv2.GaussianBlur(img_color, (7, 7), 0)
    #     edge = cv2.Canny(blur, 0, 100)
    #     edge = cv2.dilate(edge, None, iterations=1)
    #     edge = cv2.erode(edge, None, iterations=1)

    if convert_type == 'thresh':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        _, threshold = cv2.threshold(blur, 40, 120, cv2.THRESH_BINARY)
        edge = cv2.Canny(threshold, 0, 100)
        edge = cv2.dilate(edge, None, iterations=1)
        edge = cv2.erode(edge, None, iterations=1)

    return edge


def get_contours(image, convert_type):
    edge = pre_process_image(image, convert_type)
    cnts = None
    if edge is not None:
        # find contours
        cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    return cnts


def get_ratio(contours, height):
    in_per_px = 0
    if len(contours) > 0:
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue

            box = cv2.minAreaRect(c)
            box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            box = perspective.order_points(box)

            (tl, tr, br, bl) = box

            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            h_px = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            if h_px != 0:
                in_per_px = height / h_px

    return in_per_px


def show(image, contours, ratio):

    if len(contours) > 0:
        for c in contours:
            if cv2.contourArea(c) < 200:
                continue

            box = cv2.minAreaRect(c)
            box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            box = perspective.order_points(box)
            (tl, tr, br, bl) = box

            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            print(ratio)
            cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 64), 2)
            cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 255, 255), 1)

            h_px = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            height_inch = h_px * ratio
            cv2.putText(image, "{:.2f}cm".format(height_inch * 2.54),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 0, 127), 2)
