import numpy as np
import matplotlib.pyplot as plt
import time
import pyvisgraph as vg
import cv2
from matplotlib.patches import Polygon
import math



XYMIRROR = False

AREA_THRESHOLD = 500
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
MARKS_ASPECT_RATIO = 2#16/9
RED_AREA_THRESHOLD = 1000
DILATION_FACTOR = 200
GLOBAL_OBSTACLE_AREA_THRESHOLD = 1000


def getPerspectiveMatrix(centers):
    # if more than 4 objects, take the 4 extreme ones
    if len(centers) > 4:
        centers = sorted(centers, key=lambda x: (x[1], x[0]))
        centers = centers[:2] + centers[-2:]

    if len(centers) < 4:
        print("Error: 4 objects not detected")
        return

    # Sort the centers based on their position (top-left, top-right, bottom-left, bottom-right)
    centers = sorted(centers, key=lambda x: (x[1], x[0]))

    destinationWidth = IMAGE_HEIGHT * MARKS_ASPECT_RATIO
    initialX = (IMAGE_WIDTH - destinationWidth) / 2
    finalX = (IMAGE_WIDTH + destinationWidth) / 2

    # Define the destination corners of the image
    if XYMIRROR:
        dstCorners = np.array([[finalX, 0], [initialX, 0], [finalX, IMAGE_HEIGHT], [initialX, IMAGE_HEIGHT]], dtype=np.float32)
    else:
        dstCorners = np.array([[initialX, 0], [finalX, 0], [initialX, IMAGE_HEIGHT], [finalX, IMAGE_HEIGHT]], dtype=np.float32)

    srcCorners = np.array(centers, dtype=np.float32)

    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(srcCorners, dstCorners)

    return M



def correctPerspectiveStream(image,M):
    # Apply the perspective transformation to the image
    correctedImage = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return correctedImage



def findCorners(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerGreen = np.array([40, 40, 40], dtype=np.uint8)
    upperGreen = np.array([80, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lowerGreen, upperGreen)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > AREA_THRESHOLD]

    contours = [cv2.convexHull(cnt) for cnt in contours]
    
    centroids = []

    for contour in contours:

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)

    return centroids



def findGlobalObstacles(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerGreen = np.array([40, 40, 40], dtype=np.uint8)
    upperGreen = np.array([80, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lowerGreen, upperGreen)

    kernel = np.ones((DILATION_FACTOR,DILATION_FACTOR), np.uint8) 

    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) > GLOBAL_OBSTACLE_AREA_THRESHOLD]

    aproxContours = [cv2.convexHull(cnt) for cnt in contours]
    
    cv2.drawContours(frame, aproxContours, -1, (0, 255, 0), 2)

    return aproxContours



def findGoal(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerYellow = np.array([10, 100, 100], dtype=np.uint8)
    upperYellow = np.array([30, 255, 255], dtype=np.uint8)


    mask = cv2.inRange(hsv, lowerYellow, upperYellow)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    contours = [cv2.convexHull(cnt) for cnt in contours]
    contours = [c for c in contours if cv2.contourArea(c) > AREA_THRESHOLD]

    
    #keep only the biggest contour
    if len(contours) > 0:
        biggestContour = max(contours, key=cv2.contourArea)
        contours = [biggestContour]
    else:
        contours = []

    centroids = []

    for contour in contours:

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)

    centroids = np.array(centroids)
    return centroids[0]



def findThymio(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerRed = np.array([0, 100, 100], dtype=np.uint8)
    upperRed = np.array([10, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lowerRed, upperRed)

    lowerRed = np.array([160, 100, 100], dtype=np.uint8)
    upperRed = np.array([180, 255, 255], dtype=np.uint8)

    mask2 = cv2.inRange(hsv, lowerRed, upperRed)

    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) > RED_AREA_THRESHOLD]

    contours = sorted(contours, key=cv2.contourArea)

    if len(contours) != 2:
        return np.array([-1,-1]),0,0

    centroids = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.drawContours(frame, [contour], 0, (255, 255, 255), 2)


    centerPoint = np.array([centroids[1][0],centroids[1][1]])
    cv2.circle(frame, (int(centerPoint[0]), int(centerPoint[1])), 5, (0, 0, 255), -1)

    if len(centroids) == 2:
        angle = -np.arctan2(centroids[0][1] - centroids[1][1], centroids[0][0] - centroids[1][0])

    #normalize the angle between 0 and 2pi
    if angle < 0:
        angle = angle + 2*np.pi
    
    if angle > 2*np.pi:
        angle = angle - 2*np.pi

    return centerPoint, angle, centroids