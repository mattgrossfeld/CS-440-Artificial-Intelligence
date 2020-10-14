
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
    armLimits = arm.getArmLimit()
    alphaLimits = armLimits[0]
    betaLimits = armLimits[1]
    armAngles = arm.getArmAngle()
    alphaInitial = armAngles[0]
    betaInitial = armAngles[1]

    rows = int(((alphaLimits[1] - alphaLimits[0])/granularity) + 1)
    cols = int(((betaLimits[1] - betaLimits[0])/granularity) + 1)
    mazeMap = []
    for i in range(rows):
        mazeMap.append([])
        for j in range(cols):
            mazeMap[i].append(' ') # Space char = ' '
    # mazeMap is now row*cols sized 2D array filled with spaces. Initial setup
    # Now, set the map appropriately
    i = 0
    alpha = alphaLimits[0]
    for alpha in range(alphaLimits[0], alphaLimits[1], granularity):
        j = 0
        flag = 0
        beta = betaLimits[0]
        for beta in range(betaLimits[0], betaLimits[1], granularity):
            if flag != 0:
                mazeMap[i][j] = '%'
                j = j + 1
                continue
            angle = (alpha, beta)
            arm.setArmAngle(angle)
            armPos = arm.getArmPos()
            if isArmWithinWindow(armPos, window) == False:
                mazeMap[i][j] = '%'
            elif alpha == alphaInitial and beta == betaInitial:
                mazeMap[i][j] = 'P'
            elif doesArmTouchGoals(arm.getEnd(), goals) == True:
                mazeMap[i][j] = '.'
            elif doesArmTouchObstacles(armPos[:-1], obstacles) == True:
                flag = 1
                mazeMap[i][j] = '%'
            elif doesArmTouchObstacles(armPos, obstacles) == True:
                mazeMap[i][j] = '%'
            j = j + 1
        i = i + 1
    return Maze(mazeMap, [alphaLimits[0], betaLimits[0]], granularity)
