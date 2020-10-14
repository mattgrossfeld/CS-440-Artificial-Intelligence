# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position of the arm link, (x-coordinate, y-coordinate)
    """
    return (start[0] + (length * math.cos(math.radians(angle))), start[1] - (length * math.sin(math.radians(angle))))
    #pass


# References the following code discussion's answer:
# https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
def doesArmTouchObstacles(armPos, obstacles):
    """Determine whether the given arm links touch obstacles

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            obstacles (list): x-, y- coordinate and radius of obstacles [(x, y, r)]

        Return:
            True if touched. False it not.
    """    
    for arm in armPos: # Go through each arm
        armStart = np.array(arm[0])  # starting coord
        armEnd = np.array(arm[1])  # ending coord
        for obstacle in obstacles:
            r = obstacle[2]
            Q = np.array(obstacle[:-1]) # Obstacle's x,y coords
            a = np.dot(armEnd - armStart, armEnd - armStart)
            b = 2 * np.dot(armEnd - armStart, armStart - Q)
            c = np.dot(armStart, armStart) + np.dot(Q, Q) - 2 * np.dot(armStart, Q) - r**2
            d = b**2 - 4*a*c
            if d < 0:
                continue
            sd = math.sqrt(d)
            solution1 = (-b + sd) / (2*a)
            solution2 = (-b - sd) / (2*a)
            if (0 <= solution1 and solution1 <= 1) or (0 <= solution2 and solution2 <= 1):
                return True
    return False    



def doesArmTouchGoals(armEnd, goals):
    """Determine whether the given arm links touch goals

        Args:
            armEnd (tuple): the arm tick position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]

        Return:
            True if touched. False it not.
    """
    armX = armEnd[0]
    armY = armEnd[1]

    for goal in goals:
        r = goal[2]
        d = math.sqrt((armX - goal[0])**2 + (armY - goal[1])**2)
        if (d <= r):
            return True
    return False

def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False it not.
    """
    for arm in armPos:
        armStart = arm[0]
        armEnd = arm[1]
        if (armStart[0] > window[0]) or (armStart[0] < 0) or (armStart[1] > window[1]) or (armStart[1] < 0):
            return False
        if (armEnd[0] > window[0]) or (armEnd[0] < 0) or (armEnd[1] > window[1]) or (armEnd[1] < 0):
            return False
    return True