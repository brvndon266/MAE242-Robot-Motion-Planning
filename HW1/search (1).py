#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sonia martinez
"""

# Please do not distribute or publish solutions to this
# exercise. You are free to use these problems for educational purposes, please refer to the source.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from mazemods import maze
from mazemods import makeMaze
from mazemods import collisionCheck
from mazemods import makePath
from mazemods import getPathFromActions
from mazemods import getCostOfActions
from mazemods import stayWestCost
from mazemods import stayEastCost




def depthFirstSearch(xI,xG,n,m,O):
  """
  Search the deepest nodes in the search tree first.
 
  Your search algorithm needs to return a list of actions
  and a path that reaches the goal.  
  Make sure to implement a graph search algorithm.
  Your algorithm also needs to return the cost of the path. 
  Use the getCostOfActions function to do this.
  Finally, the algorithm should return the number of visited
  nodes in your search.
 
  """
  "*** YOUR CODE HERE ***"
  stack= [xI]
  visited=set()
  visited.add(xI)
  parent = {xI: None}
  parentAction = {xI:None}
  U = [(1,0),(-1,0),(0,1),(0,-1)]  # legal controls
  while stack:
    s= stack.pop()
    if s ==xG:
        actions = []
        curr = s
        while parentAction[curr] is not None:
          actions.append(parentAction[curr])
          curr=parent[curr]
        actions.reverse()
        cost= getCostOfActions(xI,actions,O)
        return actions, cost, len(visited)
    
    for u in U:
         if not collisionCheck(s, u, O):
             sNext = (s[0] + u[0], s[1] + u[1])
             if sNext not in visited:
              visited.add(sNext)
              parent[sNext] = s
              parentAction[sNext] = u
              stack.append(sNext)
  return [], 0, len(visited)  # FAILURE
  
def breadthFirstSearch(xI,xG,n,m,O):
  """
  Search the shallowest nodes in the search tree first [p 85].
 
  Your search algorithm needs to return a list of actions
  and a path that reaches the goal. Make sure to implement a graph 
  search algorithm.
  Your algorithm also needs to return the cost of the path. 
  Use the getCostOfActions function to do this.
  Finally, the algorithm should return the number of visited
  nodes in your search.

  """
  "*** YOUR CODE HERE ***"
  queue= [xI]
  visited=set()
  visited.add(xI)
  parent = {xI: None}
  parentAction = {xI:None}
  U = [(1,0),(-1,0),(0,1),(0,-1)]  # legal controls
  while queue:
    s= queue.pop(0)
    if s ==xG:
        actions = []
        curr = s
        while parentAction[curr] is not None:
          actions.append(parentAction[curr])
          curr=parent[curr]
        actions.reverse()
        cost= getCostOfActions(xI,actions,O)
        return actions, cost, len(visited)
    
    for u in U:
         if not collisionCheck(s, u, O):
             sNext = (s[0] + u[0], s[1] + u[1])
             if sNext not in visited:
              visited.add(sNext)
              parent[sNext] = s
              parentAction[sNext] = u
              queue.append(sNext)
  return [], 0, len(visited)  # FAILURE
  

def DijkstraSearch(xI,xG,n,m,O,cost=stayWestCost):
  """
  Search the nodes with least cost first. 
  
  Your search algorithm needs to return a list of actions
  and a path that reaches the goal. Make sure to implement a graph 
  search algorithm.
  Your algorithm also needs to return the total cost of the path using
  either the stayWestCost or stayEastCost function.
  Finally, the algorithm should return the number of visited
  nodes in your search.
  """
  "*** YOUR CODE HERE ***"
  frontier= [(0,xI)] # Cost and State List
  visited=set()
  g={xI:0}
  parent = {xI: None}
  parentAction = {xI:None}
  U = [(1,0),(-1,0),(0,1),(0,-1)]  # legal controls
  while frontier:
    min_idx = 0
    for i in range(len(frontier)):
       if frontier[i][0] < frontier[min_idx][0]:
          min_idx=i
    cost_s,s=frontier.pop(min_idx)
    if s in visited:
       continue
    visited.add(s)
    if s == xG:        # same as BFS
        actions = []
        curr = s
        while parentAction[curr] is not None:
            actions.append(parentAction[curr])
            curr = parent[curr]
        actions.reverse()
        cost = getCostOfActions(xI, actions, O)
        return actions, g[xG], len(visited)

    for u in U:
        if not collisionCheck(s, u, O):
            sNext = (s[0] + u[0], s[1] + u[1])
            new_cost = g[s] +cost(s,[u],O)    # step 13: compute cost to reach sNext
            if sNext not in g or new_cost < g[sNext]:  # step 13: only update if cheaper
                g[sNext] = new_cost
                parent[sNext] = s
                parentAction[sNext] = u
                frontier.append((new_cost, sNext))     # CHANGED: store cost with state

  return [], 0, len(visited)
  


def nullHeuristic(state,goal):
   """
   A heuristic function estimates the cost from the current state to the nearest
   goal.  This heuristic is trivial.

   """
   return 0

def manhattanHeuristic(state,goal):
   """
   A heuristic function estimates the cost from the current state to the nearest
   goal.  This heuristic is trivial.

   """
   return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def aStarSearch(xI,xG,n,m,O,heuristic=manhattanHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  """The function uses a function heuristic as an argument. We have used
  the null heuristic here first, you should redefine heuristics as part of 
  the homework. 
  Your algorithm also needs to return the total cost of the path using
  getCostofActions functions. 
  Finally, the algorithm should return the number of visited
  nodes during the search."""
  "*** YOUR CODE HERE ***"
  frontier= [(0,xI)] # Cost and State List
  visited=set()
  g={xI:0}
  parent = {xI: None}
  parentAction = {xI:None}
  U = [(1,0),(-1,0),(0,1),(0,-1)]  # legal controls
  while frontier:
    min_idx = 0
    for i in range(len(frontier)):
       if frontier[i][0] < frontier[min_idx][0]:
          min_idx=i
    cost_s,s=frontier.pop(min_idx)
    if s in visited:
       continue
    visited.add(s)
    if s == xG:        # same as BFS
        actions = []
        curr = s
        while parentAction[curr] is not None:
            actions.append(parentAction[curr])
            curr = parent[curr]
        actions.reverse()
        return actions, g[xG], len(visited)

    for u in U:
        if not collisionCheck(s, u, O):
            sNext = (s[0] + u[0], s[1] + u[1])
            new_cost = g[s] + 1    # step 13: compute cost to reach sNext
            if sNext not in g or new_cost < g[sNext]:  # step 13: only update if cheaper
                g[sNext] = new_cost
                parent[sNext] = s
                parentAction[sNext] = u
                frontier.append((new_cost + heuristic(sNext,xG), sNext))     # CHANGED: store cost with state

  return [], 0, len(visited)

  

    
# Plots the path
def showPath(xI,xG,path,n,m,O):
    gridpath = makePath(xI,xG,path,n,m,O)
    fig, ax = plt.subplots(1, 1) # make a figure + axes
    ax.imshow(gridpath) # Plot it
    ax.invert_yaxis() # Needed so that bottom left is (0,0)
     
if __name__ == '__main__':
    # Run test using smallMaze.py (loads n,m,O)
    #from smallMaze import *
    from mediumMaze import *  # try these mazes too
    #from bigMaze import *     # try these mazes too
    #maze(n,m,O) # prints the maze
    
    # Sample collision check
    #x, u = (5,4), (1,0)
    #testObs = [[6,6,4,4]]
    #collided = collisionCheck(x,u,testObs)
    #print('Collision!' if collided else 'No collision!')
    
    # Sample path plotted to goal
    xI = (1,1)
    xG = (20,2)
    #actions = [(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(0,1),
    #           (1,0),(1,0),(1,0),(0,-1),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0)]
    #path = getPathFromActions(xI,actions)
    #showPath(xI,xG,path,n,m,O)
    
    #Call DFS Alg
    actions_dfs, cost_dfs, nodes_dfs = depthFirstSearch(xI, xG, n, m, O)
    #print('DFS Actions:', actions_dfs)
    #print('DFS Cost:', cost_dfs)
    #print('Nodes Explored:', nodes_dfs)
    path_dfs = getPathFromActions(xI, actions_dfs)
    showPath(xI, xG, path_dfs, n, m, O)
    plt.title("DFS Path")
    
    #Call BFS Alg
    actions_bfs, cost_bfs, nodes_bfs = breadthFirstSearch(xI, xG, n, m, O)
    #print('DFS Actions:', actions_bfs)
    #print('DFS Cost:', cost_bfs)
    #print('Nodes Explored:', nodes_bfs)
    path_bfs = getPathFromActions(xI, actions_bfs)
    showPath(xI, xG, path_bfs, n, m, O)
    plt.title("BFS Path")
    
    # Dijkstra with west cost
    actions_dijk_w, cost_dijk_w, nodes_dijk_w = DijkstraSearch(xI, xG, n, m, O, stayWestCost)
    #print('Dijkstra West Cost:', cost_dijk_w)
    #print('Dijkstra West Nodes:', nodes_dijk_w)
    path_dijk_w = getPathFromActions(xI, actions_dijk_w)
    showPath(xI, xG, path_dijk_w, n, m, O)
    plt.title("Dijkstra West Cost Path")

    # Dijkstra with east cost
    actions_dijk_e, cost_dijk_e, nodes_dijk_e = DijkstraSearch(xI, xG, n, m, O, stayEastCost)
    #print('Dijkstra East Cost:', cost_dijk_e)
    #print('Dijkstra East Nodes:', nodes_dijk_e)
    path_dijk_e = getPathFromActions(xI, actions_dijk_e)
    showPath(xI, xG, path_dijk_e, n, m, O)
    plt.title("Dijkstra East Cost Path")

    # A*
    actions_astar, cost_astar, nodes_astar = aStarSearch(xI, xG, n, m, O, manhattanHeuristic)
    path_Astar=getPathFromActions(xI, actions_astar)
    showPath(xI, xG, path_Astar, n, m, O)
    plt.title("A^ Cost Path")
    plt.show()
    
    # Cost of that path with various cost functions
    simplecost = getCostOfActions(xI,actions_bfs,O)
    westcost = stayWestCost(xI,actions_bfs,O)
    eastcost = stayEastCost(xI,actions_bfs,O)
    print('Basic cost was %d, stay west cost was %d, stay east cost was %d' %
          (simplecost,westcost,eastcost))
    

    # Cost of DFS
    simplecostdfs = getCostOfActions(xI,actions_dfs,O)
    westcostdfs = stayWestCost(xI,actions_dfs,O)
    eastcostdfs = stayEastCost(xI,actions_dfs,O)
    print('Basic cost was %d, stay west cost was %d, stay east cost was %d' %
          (simplecostdfs,westcostdfs,eastcostdfs))
    
    
    
