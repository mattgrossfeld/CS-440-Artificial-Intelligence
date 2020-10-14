# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)
from queue import PriorityQueue as pq
import math

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "astar": astar,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    objectives = maze.getObjectives()   # Just get the objectives first to check edge case (start == goal)
    frontier = []                       # Empty queue siince it is BFS
    start = maze.getStart()              # Starting point
    frontier.append([start])               # Add the start node to frontier
    if (start in objectives):              # If our start is the goal, return an empty frontier.
        return frontier
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
                    return newPath
            explored.append(node)
    return []

def dfs(maze):
    """
    Runs DFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    objectives = maze.getObjectives()   # Just get the objectives first to check edge case (start == goal)
    frontier = []                       # Empty stack siince it is DFS
    start = maze.getStart()              # Starting point
    frontier.append(start)               # Add the start node to frontier

    if (objectives[0] == start):              # If our start is the goal, return an empty frontier.
        return frontier

    predecessors = {}                       # Dictionary of predecessors to keep track of our path.
    explored = []                           # Nodes explored
    
    while len(frontier) != 0:            # Frontier cannot be empty
        node = frontier.pop()               # Pop the last value. Stack so last in = first out.
        explored.append(node)           # Have now visited the current node
        neighbors = maze.getNeighbors(node[0],node[1])  # Get its neighbors
        for neighbor in neighbors:  # For each neighbor (four neighbors max)
            if (neighbor not in frontier) and (neighbor not in explored) and (maze.isValidMove(neighbor[0], neighbor[1])): # If neighbor hasn't been explored & is valid
                frontier.append(neighbor)
                predecessors[neighbor] = node
    path = []
    goal = objectives[0]
    path.insert(0, goal)
    while (goal != start):
        path.insert(0, predecessors[goal])
        goal = predecessors[goal]
    return path

# Heuristic = manhattan distance
def heuristic(node1, node2):
    manhattan = abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
    return manhattan


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    objectives = maze.getObjectives()   # Just get the objectives first to check edge case (start == goal)
    frontier = []                       # Empty queue since it is A*
    start = maze.getStart()              # Starting point
    frontier.append(start)               # Add the start node to frontier
    if (start == objectives[0]):        # If we start at the objective, return.
        return frontier
    explored = []                       # explored states
    cost = pq()                         # Priority queue of nodes. Holds them by distance.
    cost.put((heuristic(start, objectives[0]), frontier))   
    path = []
    while (cost.empty() != True):   # While the priority queue is not empty we loop
        path = cost.get()[1]    # Get the current frontier/path
        node = path[len(path)-1]    # Get the most recent one.
        if node in explored:
            continue
        if node == objectives[0]:
            return path
        explored.append(node)          # Explored the node
        neighbors = maze.getNeighbors(node[0], node[1])    # Get the neighbors
        for neighbor in neighbors:                         # For each neighbor
            manhattanDist = heuristic(neighbor, objectives[0])    # Calculate the manhattan distance
            if (neighbor not in explored) or (neighbor not in frontier) and (maze.isValidMove(neighbor[0], neighbor[1])):
                frontier = list(path)    # Set the path as the frontier.
                frontier.append(neighbor)    # Add the neighbor to the frontier.
                cost.put((manhattanDist + len(frontier), frontier))  # Push the new cost.

# NOTE: Class was modified from geeksforgeeks.org code on Kruskal's algorithm
# Original code found here: https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
# Kruskal's algorithm used.
class MST_Graph:
    def __init__(self, nodes):  # Nodes should be objectives
        self.vertices = nodes   # Vertices of graph
        self.edges = []         # Edges of Graph. initially empty

    def addEdge(self, u, v, w):         # Adding edges
        self.edges.append([u, v, w])    # Add an edge as a list. u = node1, v = node2, w = weight of edge

    def find(self, parent, child):      # Find set of child.
        if parent[child] == child:
            return child                # If the parent of child is itself, return it.
        else:
            return self.find(parent, parent[child])     # Recursively find the parent.

    def union(self, rank, parent, u, v):
        rootU = self.find(parent, u)            # Get the root of u
        rootV = self.find(parent, v)            # Get the root of v

        if rank[rootV] < rank[rootU]:           # If root of v has a smaller rank than root of u
            parent[rootV] = rootU               # Set the parent of root V to root U
        elif rank[rootV] > rank[rootU]:         # If the root of V has a larger rank that root of u
            parent[rootU] = rootV               # Set parent of root U to root V
        else:                                   # Else, they're equally ranked and must have a tie-breaker.
            parent[rootV] = rootU               # Set root V parent to root U
            rank[rootU] += 1                    # Increment the rank of root U by one to break the tie.

    def KruskalMST(self, node):
        weights = 0                             # The weight of our MST
        nodes = self.vertices                   # The vertices of our graph
        nodes.append(node)                      # Add the node given
        cost = pq()                             # A priority queue for holding the cost of each node.
        rank = {}                               # Dictionary of ranks for each node.
        parent = {}                             # Dictionary of parents for each node.
        for u in nodes:                         # For each node u
            parent[u] = u                       # Make it its own parent for now
            rank[u] = math.inf                  # Make the rank infinity for now
            for v in nodes:                     # For each node v
                if u != v:                      
                    cost.put((heuristic(u,v),u,v))  # If u is not the same node as v, we will calculate the cost and put it in the priority queue

        while len(self.edges) < len(self.vertices)-1:   # Go through EVERY edge. |E| = |V|-1
            smallest = cost.get()                       # Get the minimum cost between two nodes u,v
            x = self.find(parent, smallest[1])          # Set x as the root of node u
            y = self.find(parent, smallest[2])          # Set y as the root of node v
            if x != y:                                  # If x and y are not the same node
                self.addEdge(smallest[1], smallest[2], heuristic(smallest[1], smallest[2])) # Add an edge between them
                self.union(rank, parent, x, y)                                              # Union them.
        for edge in self.edges:
            weights += edge[2]                                                              # Sum the edge weights.
        return weights


def remainingObjectives(objectives, objectivesReached):
    objectivesLeft = []
    if False not in objectivesReached:
        return objectivesLeft
    for index, objective in enumerate(objectives):
        if objectivesReached[index] == False:
            objectivesLeft.append(objective)
    return objectivesLeft

def getState(state, node, objectives):
    if node in objectives:
        index = objectives.index(node)
        objReached = list(state[1])
        objReached[index] = True
        objReached = tuple(objReached)
    else:
        objReached = state[1]
    return (node, objReached)

def astar_multi(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    objectives = maze.getObjectives()       # Get objectives
    
    if len(objectives) == 1:            # If we only have one objective, just do regular astar
        return astar(maze)
    
    start = maze.getStart()             # get the start

    frontier = pq()                        # Priority queue of cost + tracking stuff.

    objectivesReached = []
    for objective in objectives:
        objectivesReached.append(False)
    
    #print(objectives)
    #print(objectivesReached)

    startState = (start, tuple(objectivesReached))     # State = Node & num of Objectives left.    
    costOfState = {}
    costOfState[startState] = 0         # Initially, no cost
    frontier.put((0, (startState, [start])))   # Frontier will hold: (Priority in queue, (state, path))
    objectivesLeft = remainingObjectives(objectives, objectivesReached)
    heuristics = {}
    counter = 0
    while (frontier.empty() != True) and counter < 10**6:
        current = frontier.get()
        currentState = current[1][0]
        currentPath = current[1][1]
        currentNode = currentPath[len(currentPath)-1]
        currentCost = costOfState[currentState]

        objectivesLeft = remainingObjectives(objectives, objectivesReached)

        neighbors = maze.getNeighbors(currentNode[0],currentNode[1])
        
        for neighbor in neighbors:
            neighborState = getState(currentState, neighbor, objectives)
            #print(neighborState)
            #print(neighbor, tuple(objectivesReached))
            newCost = costOfState[currentState] + 1
            newPath = list(currentPath)
            newPath.append(neighbor)

            if False not in neighborState[1]:
                    return newPath
                
            if (neighborState not in costOfState.keys()) or (costOfState[neighborState] > newCost) and maze.isValidMove(neighbor[0],neighbor[1]):
                if neighbor in objectivesLeft:
                    index = objectives.index(neighbor)
                    objectivesReached[index] = True
                    objectivesLeft = remainingObjectives(objectives, objectivesReached)
                    break
                sortedObjectives = tuple(sorted(objectivesLeft, key=lambda tup: [tup[0], tup[1]]))
                heuristicsKey = (neighbor, sortedObjectives)
                if heuristicsKey not in heuristics.keys():
                    heuristic = MST_Graph(objectivesLeft)
                    mst_weights = heuristic.KruskalMST(neighbor)
                    heuristics[heuristicsKey] = (heuristic, mst_weights)
                    
                heuristic = heuristics[heuristicsKey][0]
                mst_weights = heuristics[heuristicsKey][1]
                priority = newCost + mst_weights
                costOfState[neighborState] = newCost
                frontier.put((priority, (neighborState, newPath)))
        counter += 1
        #print(counter)
    return newPath


def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
