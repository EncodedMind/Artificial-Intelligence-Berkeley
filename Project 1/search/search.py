# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    # print("Start:", problem.getStartState()) #(5, 5)
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState())) #False
    # print("Start's successors:", problem.getSuccessors(problem.getStartState())) #[((5, 4), 'South', 1), ((4, 5), 'West', 1)]

    stack = util.Stack()
    visited = set()

    stack.push((problem.getStartState(), []))

    while not stack.isEmpty():
        state, path = stack.pop()

        if state not in visited:
            visited.add(state)
        else:
            continue

        if problem.isGoalState(state):
            return path

        successors = problem.getSuccessors(state)

        for nextstate, direction, cost in successors:
            if nextstate not in visited:
                stack.push((nextstate, path + [direction]))

    return None

"""
The following implementation uses the recursive dfs.
It is efficient but pushes the states in the reverse order,
so it expands more nodes, as the exercise instructions state.
"""

# def depthFirstSearch(problem: SearchProblem) -> List[Directions]:

#     visited = set()

#     def pathfinder(state, path):
#         if problem.isGoalState(state):
#             return path
        
#         visited.add(state)
        
#         successors = problem.getSuccessors(state)
        
#         for nextstate, direction, cost in successors:
#             if nextstate not in visited:
#                 newpath = pathfinder(nextstate, path + [direction])
#                 if newpath:
#                     return newpath
            
#         return None
    
#     path = pathfinder(problem.getStartState(), [])
#     return path

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""

    queue = util.Queue()
    visited = set()

    queue.push((problem.getStartState(), []))

    while not queue.isEmpty():
        state, path = queue.pop()

        if state not in visited:
            visited.add(state)
        else:
            continue
        
        if problem.isGoalState(state):
            return path

        successors = problem.getSuccessors(state)

        for nextstate, direction, cost in successors:
            if nextstate not in visited:
                queue.push((nextstate, path + [direction]))

    return None

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((state, [], 0), 0)
    explored = set()
    
    while not frontier.isEmpty():        
        state, path, cost = frontier.pop()
            
        if state not in explored:
            explored.add(state)
        else:
            continue

        if problem.isGoalState(state):
            return path
        
        successors = problem.getSuccessors(state)

        for nextstate, action, stepcost in successors:
            if nextstate not in explored:
                frontier.update((nextstate, path + [action], cost + stepcost), cost + stepcost)
                
    return None

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((state, [], 0), 0 + heuristic(state, problem))
    
    """
    We are keeping track of the lowest cost to reach each state and update a
    dictionary (provides us with easier access to each state) when a lower cost is
    found. We don't need the 'explored' set because we might have to expand some
    nodes more than once, according to the information of the exercise.
    """
    bestcosts = {} 
    bestcosts[state] = 0

    while not frontier.isEmpty():        
        state, path, cost = frontier.pop()
        
        # Similar to: if state in explored: continue (A state to ignore)
        if state in bestcosts and cost > bestcosts[state]:
            continue
        
        if problem.isGoalState(state):
            return path
        
        successors = problem.getSuccessors(state)

        for nextstate, action, stepcost in successors:
            # Only consider this next state if we've found a cheaper path
            if nextstate not in bestcosts or cost + stepcost < bestcosts[nextstate]: # Similar to: if nextstate not in explored. We also need to check if the new cost is better than the existing cost of the state
                bestcosts[nextstate] = cost + stepcost  # Update the best cost
                frontier.update((nextstate, path + [action], cost + stepcost), cost + stepcost + heuristic(nextstate, problem))
                
    return None

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
