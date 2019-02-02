import sys

import numpy as np
import cv2 as cv


release = False
color = {"GOOD": (0, 255, 0), "FAIR": (255, 0, 255), "BAD": (0, 0, 255)}
# Init video stream
cap = cv.VideoCapture('../data/cookies.avi')

if not cap.isOpened():
    print("NO VIDEO AVAILABLE")
    sys.exit()

# Get the properties of video
width_f = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height_f = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
fps_src = cap.get(cv.CAP_PROP_FPS)
size_f = (int(width_f), int(height_f))
print(size_f)
# Define the codec and create output stream
codec = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('../data/temp.avi', codec, 24, size_f)

# Create the main window
cv.namedWindow('Main')

# Create the main window
cv.namedWindow('RoI')
# Create the HSV track bar
lower_bound = np.array([3, 127, 31], dtype=np.uint8)
upper_bound = np.array([15, 255, 255], dtype=np.uint8)
cv.createTrackbar('H', 'Main', lower_bound[0], 180, lambda x: None)
cv.createTrackbar('S', 'Main', lower_bound[1], 255, lambda x: None)
cv.createTrackbar('V', 'Main', lower_bound[2], 255, lambda x: None)


def is_top_clear(toprows):
    gray = cv.cvtColor(toprows, cv.COLOR_BGR2GRAY)
    retval, mask = cv.threshold(gray, 32, 255, cv.THRESH_BINARY)
    if mask.sum() < 10000:
        return True
    return False


# Evaluate ROI
def evaluator(c, r, roi, S=None):

    h, w, chan = roi.shape

    if r * r * 3.14 * 0.75 > S:
        return "BAD"

    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    lower_bound[0] = cv.getTrackbarPos('H', 'Main')
    lower_bound[1] = cv.getTrackbarPos('S', 'Main')
    lower_bound[2] = cv.getTrackbarPos('V', 'Main')

    mask = cv.inRange(hsv, lower_bound, upper_bound)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    # mask.sum()
    contours, hierarchy = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) < 20:
        return "FAIR"

    # output = cv.bitwise_and(roi, roi, mask=mask)
    # output[mask == 0] = np.array([255, 255, 255], dtype=np.uint8)
    # cv.imshow("RoI", output)

    return "GOOD"


isPreTopClear = True

while cap.isOpened():
    # Get current frame
    success, frame = cap.read()

    if success:
        # frame = cv.flip(frame, 1)

        isTopClear = is_top_clear(frame[0:100, 30:-30, :])
        if not (isTopClear and not isPreTopClear):
            isPreTopClear = isTopClear
            continue
        isPreTopClear = isTopClear

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        retval, mask = cv.threshold(gray, 7, 255, cv.THRESH_BINARY)
        mask = cv.erode(mask, None, iterations=3)
        mask = cv.dilate(mask, None, iterations=3)

        # Contours
        contours, hierarchy = cv.findContours(
            mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # find the biggest area
            c = max(contours, key=cv.contourArea)

            rx, ry, w, h = cv.boundingRect(c)

            (x, y), radius = cv.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)

            # if w > 16 and w < 512 and h > 16 and h < 512:
            if radius > 4 and radius < 256:
                roi = frame[ry:ry+h, rx:rx+w]

                # pos = cap.get(cv.CAP_PROP_POS_MSEC)
                # key_pressed=cv.waitKey(50) & 0xFF
                # filename="../data/cookies/" + str(pos) + "." + str(key_pressed) +".jpg"
                # if key_pressed == ord('g'):
                #     cv.imwrite(filename, roi)

                S = cv.contourArea(c)
                text = evaluator(center, radius, roi, S)

                # cv.rectangle(frame, (x, y), (x+w, y+h), color[text], 5)
                cv.circle(frame, center, radius, color[text], 5)
                cv.putText(frame, text, center, cv.FONT_HERSHEY_SIMPLEX,
                           1.0, color[text], 5)

        # Write to output stream
        out.write(frame)
        cv.imshow('Main', frame)
        cv.waitKey(0)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

# Release all jobs
cap.release()
out.release()

cv.destroyAllWindows()
sys.exit()
