# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # TODO: Write your code here    
    objectives = maze.getObjectives()   # Just get the objectives first to check edge case (start == goal)
    frontier = []                       # Empty queue siince it is BFS
    start = maze.getStart()              # Starting point
    frontier.append([start])               # Add the start node to frontier
    if start in objectives:              # If our start is the goal, return an empty frontier.
        return frontier, 0
    explored = []
    
    while len(frontier) != 0:            # Frontier cannot be empty
        path = frontier.pop(0)              # Pop the first value. Queue so first in = first out.
        node = path[len(path)-1]
        if node not in explored:
            neighbors = maze.getNeighbors(node[0], node[1])  # Get its neighbors
            for neighbor in neighbors:  # For each neighbor (four neighbors max)
                newPath = list(path)
                newPath.append(neighbor)
                frontier.append(newPath)
                if neighbor in objectives:
                    return newPath, 0
            explored.append(node)
    return [], 0

def dfs(maze):
    # TODO: Write your code here    
    return [], 0

def greedy(maze):
    # TODO: Write your code here    
    return [], 0

def astar(maze):
    # TODO: Write your code here    
    return [], 0