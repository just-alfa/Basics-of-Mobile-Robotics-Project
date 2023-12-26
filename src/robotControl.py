import numpy as np
import math

KIDNAPPING_THRESHOLD = 100
KIDNAPPING_TIME = 5
OBSTACLE_THRESHOLD = 200


def getDistance(robotPosition,start, end):
    x1 = start[1]
    y1 = start[0]
    x2 = end[1]
    y2 = end[0]
    x = robotPosition[1]
    y = robotPosition[0]

    # Compute the coefficients of the line equation (Ax + By + C = 0)
    A = y2 - y1
    B = x1 - x2
    C = (x2 - x1) * y1 + (y1 - y2) * x1

    # Compute the distance from the robot position to the line
    distance = abs(A * x + B * y + C) / math.sqrt(A**2 + B**2)

    # Determine the sign of the distance based on the orientation of the line
    # If the goal is to the right of the line, the distance is positive for points to the right
    # If the goal is to the left of the line, the distance is positive for points to the left
    goalOrientation = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    if goalOrientation > 0:
        distance *= -1  # Reverse the sign for points to the left of the line

    return -distance



def robotController(pointInitial,pointFinal,position,angle,dt,alignMode):

    angleOfLine = np.arctan2(pointFinal[1]-pointInitial[1],pointFinal[0]-pointInitial[0])

    if angleOfLine < 0:
        angleOfLine = angleOfLine + 2*np.pi

    angleError = angleOfLine-angle

    if angleError > np.pi:
        angleError -= 2*np.pi
    elif angleError < -np.pi:
        angleError += 2*np.pi


    Kangle = 50
    KIangle = 5

    differentialSpeed = Kangle*angleError + KIangle*angleError*dt

    distanceToLine = getDistance(position,pointInitial,pointFinal)

    Kdistance = 1

    posCorrection = Kdistance*distanceToLine

    constantSpeed = 200


    if alignMode == True:
        constantSpeed = 0
        posCorrection = 0
        posCorrection = 0

    lSpeed = -differentialSpeed - posCorrection + constantSpeed
    rSpeed = differentialSpeed + posCorrection + constantSpeed


    return lSpeed, rSpeed, distanceToLine, angleError



def checkForObstacles(horizontalSensorValues):
    #remove last 2 values
    horizontalSensorValues = horizontalSensorValues[:-2]

    for i in range(len(horizontalSensorValues)):
        value = horizontalSensorValues[i]
        if value > OBSTACLE_THRESHOLD:
            sensorId = i
            break
        else:
            sensorId = -1
    
    if sensorId == -1:
        return
    
    else:
        #check if the obstacle is on the left or on the right
        if sensorId < 3:
            leftOrRight = "left"
        else:
            leftOrRight = "right"

    return leftOrRight
    