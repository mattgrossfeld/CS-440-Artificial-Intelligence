
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    armAngles = arm.getArmAngle()       # Arm tuple
    alpha = armAngles[0]
    beta = armAngles[1]
    armLimits = arm.getArmLimit()
    alphaLimits = armLimits[0]
    betaLimits = armLimits[1]

    start = (alpha, beta)
    rows = int(((alphaLimits[1] - alphaLimits[0])/granularity) + 1)
    cols = int(((betaLimits[1] - betaLimits[0])/granularity) + 1)
    mazeMap = []
    for i in range(rows):
        mazeMap.append([])
        for j in range(cols):
            mazeMap[i].append(SPACE_CHAR) # Space char = ' '
    # mazeMap is now row*cols sized 2D array filled with spaces. Initial setup
    # Now, set the map appropriately
    for i in range(rows):
        for j in range(cols):
            a = i * granularity + alphaLimits[0]
            b = j * granularity + betaLimits[0]
            arm.setArmAngle((a,b))
            if isArmWithinWindow(arm.getArmPos(), window) == False: # Arm out of bounds
                mazeMap[i][j] = WALL_CHAR                         # WALL_CHAR = %
            elif (a,b) == start: 
                mazeMap[i][j] = START_CHAR                        # START_CHAR = P
            elif doesArmTouchObstacles(arm.getArmPos(), obstacles + goals) == True:
                mazeMap[i][j] = WALL_CHAR
            if doesArmTouchGoals(arm.getEnd(), goals) == True:  
                mazeMap[i][j] = OBJECTIVE_CHAR
    #print(mazeMap)
    return Maze(mazeMap, [alphaLimits[0], betaLimits[0]], granularity)