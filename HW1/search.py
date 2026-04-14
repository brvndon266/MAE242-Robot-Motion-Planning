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


def depthFirstSearch(xI, xG, n, m, O):
    """
    Search the deepest nodes in the search tree first.
 
    Returns:
      actions, cost, num_visited, path
    """
    actions_all = [(1,0), (-1,0), (0,1), (0,-1)]

    stack = [(xI, [])]   # (state, actions-to-reach-state)
    visited = set()
    num_visited = 0

    while stack:
        state, actions = stack.pop()

        if state in visited:
            continue

        visited.add(state)
        num_visited += 1

        if state == xG:
            cost = getCostOfActions(xI, actions, O)
            path = getPathFromActions(xI, actions)
            return actions, cost, num_visited, path

        for u in actions_all:
            next_state = (state[0] + u[0], state[1] + u[1])

            if 0 <= next_state[0] < n and 0 <= next_state[1] < m:
                if not collisionCheck(state, u, O) and next_state not in visited:
                    stack.append((next_state, actions + [u]))

    return [], 999999999, num_visited, []

def breadthFirstSearch(xI, xG, n, m, O):
    """
    Search the shallowest nodes in the search tree first.
 
    Returns:
      actions, cost, num_visited, path
    """
    from collections import deque

    actions_all = [(1,0), (-1,0), (0,1), (0,-1)]

    queue = deque([(xI, [])])   # (state, actions-to-reach-state)
    visited = set([xI])
    num_visited = 0

    while queue:
        state, actions = queue.popleft()
        num_visited += 1

        if state == xG:
            cost = getCostOfActions(xI, actions, O)
            path = getPathFromActions(xI, actions)
            return actions, cost, num_visited, path

        for u in actions_all:
            next_state = (state[0] + u[0], state[1] + u[1])

            if 0 <= next_state[0] < n and 0 <= next_state[1] < m:
                if not collisionCheck(state, u, O) and next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, actions + [u]))

    return [], 999999999, num_visited, []


def DijkstraSearch(xI, xG, n, m, O, cost=stayWestCost):
    """
    Search the nodes with least cost first.
  
    Returns:
      actions, total_cost, num_visited, path
    """
    import heapq

    actions_all = [(1,0), (-1,0), (0,1), (0,-1)]

    pq = []
    heapq.heappush(pq, (0, xI, []))   # (path-cost, state, actions)
    best_cost = {xI: 0}
    num_visited = 0

    while pq:
        curr_cost, state, actions = heapq.heappop(pq)

        if curr_cost > best_cost.get(state, float('inf')):
            continue

        num_visited += 1

        if state == xG:
            path = getPathFromActions(xI, actions)
            return actions, curr_cost, num_visited, path

        for u in actions_all:
            next_state = (state[0] + u[0], state[1] + u[1])

            if 0 <= next_state[0] < n and 0 <= next_state[1] < m:
                if not collisionCheck(state, u, O):
                    new_actions = actions + [u]
                    new_cost = cost(xI, new_actions, O)

                    if new_cost < best_cost.get(next_state, float('inf')):
                        best_cost[next_state] = new_cost
                        heapq.heappush(pq, (new_cost, next_state, new_actions))

    return [], 999999999, num_visited, []


def nullHeuristic(state,goal):
   """
   A heuristic function estimates the cost from the current state to the nearest
   goal. This heuristic is trivial.
   """
   return 0


def manhattanHeuristic(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def euclideanHeuristic(state, goal):
    return ((state[0] - goal[0])**2 + (state[1] - goal[1])**2) ** 0.5


def aStarSearch(xI, xG, n, m, O, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.

    Returns:
      actions, total_cost, num_visited, path
    """
    import heapq

    actions_all = [(1,0), (-1,0), (0,1), (0,-1)]

    pq = []
    heapq.heappush(pq, (heuristic(xI, xG), 0, xI, []))  # (f, g, state, actions)
    best_g = {xI: 0}
    num_visited = 0

    while pq:
        f, g, state, actions = heapq.heappop(pq)

        if g > best_g.get(state, float('inf')):
            continue

        num_visited += 1

        if state == xG:
            total_cost = getCostOfActions(xI, actions, O)
            path = getPathFromActions(xI, actions)
            return actions, total_cost, num_visited, path

        for u in actions_all:
            next_state = (state[0] + u[0], state[1] + u[1])

            if 0 <= next_state[0] < n and 0 <= next_state[1] < m:
                if not collisionCheck(state, u, O):
                    new_actions = actions + [u]
                    new_g = getCostOfActions(xI, new_actions, O)

                    if new_g < best_g.get(next_state, float('inf')):
                        best_g[next_state] = new_g
                        new_f = new_g + heuristic(next_state, xG)
                        heapq.heappush(pq, (new_f, new_g, next_state, new_actions))

    return [], 999999999, num_visited, []


# Plots the path
def showPath(xI, xG, path, n, m, O, title='Path'):
    gridpath = makePath(xI, xG, path, n, m, O)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(gridpath)
    ax.invert_yaxis()
    ax.set_title(title)


if __name__ == '__main__':
    # Run test using smallMaze.py (loads n,m,O)
    #from smallMaze import *
    from mediumMaze import *  # try these mazes too
    #from bigMaze import *     # try these mazes too
    #maze(n,m,O) # prints the maze
    plt.close('all')

    # Start and goal
    xI = (1, 1)
    xG = (34, 16)

    # Plot the maze first
    # maze(n, m, O)
    # plt.title('Medium Maze')

    # DFS
    actions_dfs, cost_dfs, visited_dfs, path_dfs = depthFirstSearch(xI, xG, n, m, O)
    print('DFS')
    print('  cost =', cost_dfs)
    print('  visited nodes =', visited_dfs)
    print('  action length =', len(actions_dfs))
    showPath(xI, xG, path_dfs, n, m, O, title='DFS Path')

    # BFS
    actions_bfs, cost_bfs, visited_bfs, path_bfs = breadthFirstSearch(xI, xG, n, m, O)
    print('BFS')
    print('  cost =', cost_bfs)
    print('  visited nodes =', visited_bfs)
    print('  action length =', len(actions_bfs))
    showPath(xI, xG, path_bfs, n, m, O, title='BFS Path')

    # Dijkstra stay west
    actions_dw, cost_dw, visited_dw, path_dw = DijkstraSearch(xI, xG, n, m, O, cost=stayWestCost)
    print('Dijkstra stayWestCost')
    print('  cost =', cost_dw)
    print('  visited nodes =', visited_dw)
    print('  action length =', len(actions_dw))
    showPath(xI, xG, path_dw, n, m, O, title='Dijkstra Path (Stay West Cost)')

    # Dijkstra stay east
    actions_de, cost_de, visited_de, path_de = DijkstraSearch(xI, xG, n, m, O, cost=stayEastCost)
    print('Dijkstra stayEastCost')
    print('  cost =', cost_de)
    print('  visited nodes =', visited_de)
    print('  action length =', len(actions_de))
    showPath(xI, xG, path_de, n, m, O, title='Dijkstra Path (Stay East Cost)')

    # A* Manhattan
    actions_am, cost_am, visited_am, path_am = aStarSearch(xI, xG, n, m, O, heuristic=manhattanHeuristic)
    print('A* Manhattan')
    print('  cost =', cost_am)
    print('  visited nodes =', visited_am)
    print('  action length =', len(actions_am))
    showPath(xI, xG, path_am, n, m, O, title='A* Path (Manhattan Heuristic)')

    # A* Euclidean
    actions_ae, cost_ae, visited_ae, path_ae = aStarSearch(xI, xG, n, m, O, heuristic=euclideanHeuristic)
    print('A* Euclidean')
    print('  cost =', cost_ae)
    print('  visited nodes =', visited_ae)
    print('  action length =', len(actions_ae))
    showPath(xI, xG, path_ae, n, m, O, title='A* Path (Euclidean Heuristic)')

    plt.show()