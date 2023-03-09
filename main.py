import cv2
import detecter

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
WIDTH, HEIGHT = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

while True:
    _, frame = cap.read()

    frame_ratio = frame.copy()[:, :200]
    frame_measure = frame.copy()[:, 200:]

    cnts_ratio = detecter.get_contours(frame_ratio, 'thresh')
    cnts_measure = detecter.get_contours(frame_measure, 'thresh')

    ratio = detecter.get_ratio(cnts_ratio, 11.77165)

    detecter.show(frame_measure, cnts_measure, ratio)

    detecter.show(frame_ratio, cnts_ratio, ratio)

    cv2.imshow('frame_ratio', frame_ratio)
    cv2.imshow('frame_measure', frame_measure)
    cv2.waitKey(1)
