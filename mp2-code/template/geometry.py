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


# References the following code:
# http://www.jeffreythompson.org/collision-detection/line-circle.php
# Note that the original code is in (what appears to be) C. Code has been modified to work in Python.
# All functions have been removed and put into the single function below to keep everything together.

def doesArmTouchObstacles(armPos, obstacles):
	"""Determine whether the given arm links touch obstacles

		Args:
			armPos (list): start and end position of all arm links [(start, end)]
			obstacles (list): x-, y- coordinate and radius of obstacles [(x, y, r)]

		Return:
			True if touched. False it not.
	"""    
	for arm in armPos: # Go through each arm
		armStart = arm[0]  # starting coord
		armEnd = arm[1]  # ending coord
		for obstacle in obstacles:
			obstacleX = obstacle[0]	# x coord
			obstacleY = obstacle[1] # y coord
			r = obstacle[2]			# radius
			dStart = math.sqrt((armStart[0] - obstacleX)**2 + (armStart[1] - obstacleY)**2) # Get the start point and compare it to the obstacle
			dEnd = math.sqrt((armEnd[0] - obstacleX)**2 + (armEnd[1] - obstacleY)**2)		# Compare the end point with the obstacle
			if (dStart <= r or dEnd <= r):			# Check if the endpoints touch
				return True							# Return true if they do
			point = ((obstacleX - armStart[0])*(armEnd[0] - armStart[0]) + (obstacleY - armStart[1])*(armEnd[1]-armStart[1])) # Get the dot product
			point = point / (math.sqrt((armEnd[0] - armStart[0])**2 + (armEnd[1] - armStart[1])**2)**2)						# divide it by length**2
			closestX = armStart[0] + point*(armEnd[0] - armStart[0])		# Closest X coord
			closestY = armStart[1] + point*(armEnd[1] - armStart[1])		# Closest Y coord
			#print(closestX)
			#print(closestY)
			if (min(armStart[0], armEnd[0]) <= closestX and max(armStart[0], armEnd[0]) >= closestX): # Make sure it is between the start, end points
				if (min(armStart[1], armEnd[1]) <= closestY and max(armStart[1], armEnd[1] >= closestY)):
					if (math.sqrt((closestX - obstacleX)**2 + (closestY - obstacleY)**2)) <= r:
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