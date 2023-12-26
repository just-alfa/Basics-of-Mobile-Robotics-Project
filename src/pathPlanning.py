import numpy as np
import matplotlib.pyplot as plt
import pyvisgraph as vg
from matplotlib.patches import Polygon


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

MINIMAL_DISATNCE_BETWEEN_POINTS = 60


def convert2PyvisgraphFormat(opencvPolygons):
    pyvisPolygons = []
    for opencvPolygon in opencvPolygons:
        pyvisPolygon = [vg.Point(point[0], point[1]) for point in opencvPolygon]
        pyvisPolygons.append(pyvisPolygon)
    return pyvisPolygons



def buildGraph(cv2Polygons,position,goal):

    FAR_AWAY_PLUS = 5000
    FAR_AWAY_MINUS = -2000

    position = np.array(position)
    goal = np.array(goal)

    #position[1] = 1080 - position[1]
    goal[1] = IMAGE_HEIGHT - goal[1]

    cv2Polygons = [polygon[:, 0, :] for polygon in cv2Polygons]


    # Flip the y axis for all polygons
    for polygon in cv2Polygons:
        for point in polygon:
            point[1] = IMAGE_HEIGHT - point[1]


    #if point is in the edge, push it far far away for it not to be considered
    for polygon in cv2Polygons:
        for point in polygon:
            if point[0] <= 0:
                point[0] = FAR_AWAY_MINUS
            if point[0] >= IMAGE_WIDTH:
                point[0] = FAR_AWAY_PLUS
            if point[1] <= 0:
                point[1] = FAR_AWAY_MINUS
            if point[1] >= IMAGE_HEIGHT:
                point[1] = FAR_AWAY_PLUS      

    pyvisPolygons = convert2PyvisgraphFormat(cv2Polygons)

    # Build the visibility graph
    graph = vg.VisGraph()
    graph.build(pyvisPolygons)

    # Find the shortest path
    startPoint = vg.Point(position[0], position[1])
    endPoint = vg.Point(goal[0], goal[1])
    shortestPath = graph.shortest_path(startPoint, endPoint)

    shortestPath = np.array([[point.x, point.y] for point in shortestPath])
    #check if distance between points is too small and remove unnecessary points
    i = 0
    while i < len(shortestPath)-1:
        if np.linalg.norm(shortestPath[i]-shortestPath[i+1]) < MINIMAL_DISATNCE_BETWEEN_POINTS:
            shortestPath = np.delete(shortestPath,i+1,0)
        else:
            i += 1



    # Plotting the polygons and the shortest path
    fig, ax = plt.subplots()

    # Plot polygons
    for cv2Polygon in cv2Polygons:
        poly = Polygon(cv2Polygon, fill=None, edgecolor='b')
        ax.add_patch(poly)

    # Plot all the point in the shortest path
    ax.scatter(shortestPath[:, 0], shortestPath[:, 1], c='r')
    
    # Set axis limits
    ax.set_xlim([0, IMAGE_WIDTH])
    ax.set_ylim([0, IMAGE_HEIGHT])

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

    return shortestPath