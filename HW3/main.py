#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:12:40 2022

@author: sonia, parth, yang
"""

import numpy as np
import matplotlib.pyplot as plt

from grids import SmallGrid, MediumGrid, MediumGridBridge

############################################################################
# Part 1: In this first part, we show you how to calculate the
# transition probabilities for a given grid and error
# probability. Then we show how to compute the expected value of
# taking an action `a` in state `s`. Both of these functions will be
# helpful in implementing value iteration and policy iteration.
############################################################################


def transitionMatrix(grid, eta):
    """Compute the transition matrix `P` for all states in `grid` with
    action error rate `eta`.

    `P` is a numpy array, indexed such that `P[s_p][s][a]` is the
    probability of transitioning to `s_p` from `s` under action `a`.

    """

    # Transition matrix will be indexed by ((i_s_p, j_s_p), (i_s, j_s), a)
    P = np.zeros((grid.n, grid.m, grid.n, grid.m, grid.n_a))

    # Note that numpy allows different styles of indexing. If we
    # define the tuples s_p = (i_s_p, j_s_p) and s = (i_s, j_s), then
    # the expression P[s_p][s] is equivalent to P[i_s_p][j_s_p][i_s][j_s].
    # We use this to make the following code more readable and concise.

    for s in grid.states:
        # Assign the transition probabilities for each input
        for a in range(grid.n_a):
            # Nominal case, no error is introduced.
            s_p = grid.nextState(s, grid.u_set[a])
            P[s_p][s][a] += 1 - eta

            # Error case, rotation to the left
            s_p = grid.nextState(s, grid.u_set[(a + 1) % grid.n_a])
            P[s_p][s][a] += eta / 2

            # Error case, rotation to the right
            s_p = grid.nextState(s, grid.u_set[(a - 1) % grid.n_a])
            P[s_p][s][a] += eta / 2

    return P


def expectedValue(P, V, s, a, gamma, grid):
    """Compute the expected value of taking an action from a given state.

    Inputs:
    `P`: Matrix of transition probabilities
    `V`: Value function (e.g., the estimate from the current iteration)
    `s`: State
    `a`: Action
    `gamma`: Discount factor
    `grid`: The grid world

    Outputs: The inner term of the Bellman operator, i.e., sum over
    all s' of P(s' | s, a)*(R(s, a, s') + gamma*V(s')). Note that in
    this problem the reward only depends on s' and is the negative of
    the cost.

    """
    # Get the set of all possible next states
    s_primes = set(grid.nextState(s, u) for u in grid.u_set)

    # Sum over them to compute the expectation
    return sum(P[s_p][s][a] * (-grid.cost(s_p) + gamma*V[s_p])
               for s_p in s_primes)


############################################################################
# Part 2: Now it's time to implement value iteration and policy
# iteration. We have provided templates that show the basic structure
# your code should follow.
############################################################################


def valueIteration(gamma, eta, grid):
    """Implement value iteration with a discount factor `gamma` and
    noise probability `eta`.

    Outputs:
    `V`: Value function, numpy array of (n,m) dimensions
    `pi`: Policy, numpy array of (n,m) dimensions

    """
    P = transitionMatrix(grid, eta)

    tolerance = 1e-3                            # Convergence error
    V = np.zeros((grid.n, grid.m))              # Value function
    pi = np.zeros((grid.n, grid.m), dtype=int)  # Policy
    iterations = 0                              # No. of iterations

    # ------- Your code goes here -------
    while True:
        delta = 0
        V_old = V.copy()

        for s in grid.states:
            v_old = V_old[s]

            values = [
                expectedValue(P, V_old, s, a, gamma, grid)
                for a in range(grid.n_a)
            ]

            V[s] = max(values)
            pi[s] = int(np.argmax(values))

            delta = max(delta, abs(v_old - V[s]))

        iterations += 1

        if delta < tolerance:
            break

    return V, pi, iterations


def policyIteration(gamma, eta, grid):
    """Implement (offline) policy iteration with a discount factor
    `gamma` and noise probability `eta`.

    Output:
    `V`: Value function, numpy array of (n,m) dimensions
    `pi`: Policy, numpy array of (n,m) dimensions

    """
    P = transitionMatrix(grid, eta)

    tolerance = 1e-3                            # Tolerance for evaluation
    V = np.zeros([grid.n, grid.m])              # Value function
    pi = np.zeros([grid.n, grid.m], dtype=int)  # Policy
    i_imprv = 0                                 # No. of improvement iterations
    i_eval = 0                                  # No. of evaluation iterations

    # ------- Your code goes here -------
    while True:

        # ---- Policy Evaluation ----
        delta = np.inf

        while delta >= tolerance:
            delta = 0
            V_old = V.copy()

            for s in grid.states:
                v_old = V_old[s]

                V[s] = expectedValue(P, V_old, s, pi[s], gamma, grid)

                delta = max(delta, abs(v_old - V[s]))

            i_eval += 1

        # ---- Policy Improvement ----
        policy_stable = True

        for s in grid.states:
            add_action = pi[s]

            values = [
                expectedValue(P, V, s, a, gamma, grid)
                for a in range(grid.n_a)
            ]

            pi[s] = int(np.argmax(values))

            if add_action != pi[s]:
                policy_stable = False

        i_imprv += 1

        if policy_stable:
            break

    return V, pi, i_eval, i_imprv


############################################################################
# Part 3: Here is where you will run your value and policy iteration
# algorithms, following the instructions from the pdf. We have
# provided an outline of what functions you might call to help you
# answer the questions.
############################################################################


# Parameters for each subproblem, stored as [gamma, eta]
params = {
    'a': [0.9, 0.2],
    'b': [0.9, 0.2],
    # For parts c and d, fill in the values of gamma and eta that
    # result in the desired behavior
    'c': [0.5, 0.1],
    'd1': [0.4, 0.02], # Close exit, risking the cliff
    'd2': [0.3, 0.2], # Close exit, avoiding the cliff
    'd3': [0.9, 0.2], # Far exit, risking the cliff
    'd4': [0.9, 0.5], # Far exit, avoiding the cliff
}

if __name__ == '__main__':

    # ---------- Part (a) ----------- #
    smallGrid = SmallGrid()

    gamma, eta = params['a']
    V, pi, i_eval, i_imprv = policyIteration(gamma, eta, smallGrid)

    print(f"Part (a): {i_eval} evaluations and {i_imprv} improvements.")

    # ---------- Part (b) ----------- #

    gamma, eta = params['b']
    V, pi, iterations = valueIteration(gamma, eta, smallGrid)

    print(f"Part (b): {iterations} iterations.")

    # ---------- Part (c) ----------- #
    # Change the values in `params` above to get the desired behavior
    mediumGrid = MediumGrid()

    gamma, eta = params['c']
    V, pi, _ = valueIteration(gamma, eta, mediumGrid)

    print(f"Part (c): gamma: {gamma} , eta: {eta}.")

    # Add these to verify:
    #mediumGrid.printPolicy(pi)
    # mediumGrid.printValues(V)
    
    path = mediumGrid.getNominalPathFromPolicy(pi)
    mediumGrid.plotPath(path, show=True)
    print(path)

    # ---------- Part (d) ----------- #
    # Change the values in `params` above to get the desired behavior
    mediumGridBridge = MediumGridBridge()

    # 1) Close exit, risking the cliff
    gamma, eta = params['d1']
    print(f"Part (d) 1): gamma: {gamma}, eta: {eta}.")
    V, pi, _ = valueIteration(gamma, eta, mediumGridBridge)
    
    path = mediumGrid.getNominalPathFromPolicy(pi)
    mediumGrid.plotPath(path, show=True)
    print(path)

    # 2) Close exit, avoiding the cliff
    gamma, eta = params['d2']
    print(f"Part (d) 2): gamma: {gamma}, eta: {eta}.")
    V, pi, _ = valueIteration(gamma, eta, mediumGridBridge)
    
    path = mediumGrid.getNominalPathFromPolicy(pi)
    mediumGrid.plotPath(path, show=True)
    print(path)
    
    # 3) Far exit, risking the cliff
    gamma, eta = params['d3']
    print(f"Part (d) 3): gamma: {gamma}, eta: {eta}.")
    V, pi, _ = valueIteration(gamma, eta, mediumGridBridge)
    
    path = mediumGrid.getNominalPathFromPolicy(pi)
    mediumGrid.plotPath(path, show=True)
    print(path)

    # 4) Far exit, avoiding the cliff
    gamma, eta = params['d4']
    print(f"Part (d) 4): gamma: {gamma}, eta: {eta}.")
    V, pi, _ = valueIteration(gamma, eta, mediumGridBridge)

    path = mediumGrid.getNominalPathFromPolicy(pi)
    mediumGrid.plotPath(path, show=True)
    print(path)
    
############################################################################
# Part 4: In this final part we demonstrate some of the plotting and
# printing functions available for visualizing and debugging the
# results of your algorithms.
############################################################################

    if False:  # Change this to true to see the output
        # Example policy and value function
        V = np.random.rand(mediumGrid.n, mediumGrid.m)
        pi = np.ones((mediumGrid.n, mediumGrid.m), dtype=int)  # Always up

        # Plot the empty grid with start, goal, and penalty locations marked
        # To plot on the small grid instead, use smallGrid.plot
        mediumGrid.plot(show=True)

        # Print a value function or policy to the terminal
        mediumGrid.printValues(V)
        mediumGrid.printPolicy(pi)

        # Plot the value function (red is low, green is high)
        mediumGrid.plotValues(V, show=True)

        # Plot a noise-free path from the start node under policy pi
        # Note that these paths won't look good until you compute a good policy
        path = mediumGrid.getNominalPathFromPolicy(pi)
        mediumGrid.plotPath(path, show=True)

        # Plot a noisy path from the start node under policy pi
        path = mediumGrid.getRandomPathFromPolicy(pi, eta)
        mediumGrid.plotPath(path, show=True)