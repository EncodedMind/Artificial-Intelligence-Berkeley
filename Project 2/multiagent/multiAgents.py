# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Take into consideration the successorGameState.getScore()
        score = successorGameState.getScore()

        # The closer Pacman is to food, the better the state should be

        fooddists = []
        for food in newFood.asList():
            fooddists.append(manhattanDistance(newPos, food))

        distance_to_nearest_food = min(fooddists) if fooddists else 0
        
        if distance_to_nearest_food != 0:
            score += 1 / distance_to_nearest_food
        else:
            score += 1000

        # If a ghost is close to Pacman, the state should be worse
        # Unless it is scared, where Pacman could move towards it
        
        ghostdists = []
        scaredghostdists = []
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            if ghostState.scaredTimer == 0:
                ghostdists.append(manhattanDistance(newPos, ghostPos))
            else:
                scaredghostdists.append(manhattanDistance(newPos, ghostPos))

        distance_from_ghost = min(ghostdists) if ghostdists else 0
        distance_from_scared_ghost = min(scaredghostdists) if scaredghostdists else 0

        if distance_from_ghost != 0:
            score -= 1 / distance_from_ghost
        else:
            score -= 1000
        score += distance_from_scared_ghost
        
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """            

        def maxvalue(state, agentIndex, depth): # returns a utility value
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = float('-inf')
            for a in state.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v = max(v, minvalue(state.generateSuccessor(agentIndex, a), 0, depth + 1))
                else:
                    v = max(v, minvalue(state.generateSuccessor(agentIndex, a), agentIndex + 1, depth))
            
            return v

        def minvalue(state, agentIndex, depth): # returns a utility value
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = float('inf')
            for a in state.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v = min(v, maxvalue(state.generateSuccessor(agentIndex, a), 0, depth + 1))
                else:
                    v = min(v, minvalue(state.generateSuccessor(agentIndex, a), agentIndex + 1, depth))
            
            return v
        
        agentIndex = 0 # Pacman plays first
        depth = 0

        bestscore = float('-inf')
        for a in gameState.getLegalActions(agentIndex):
            score = minvalue(gameState.generateSuccessor(agentIndex, a), agentIndex + 1, depth)
            if score > bestscore:
                bestscore = score
                bestaction = a

        return bestaction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def maxvalue(state, a, b, agentIndex, depth): # returns a utility value
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = float('-inf')
            for action in state.getLegalActions():
                if agentIndex == gameState.getNumAgents() - 1:
                    value = minvalue(state.generateSuccessor(agentIndex, action), a, b, 0, depth + 1)
                else:
                    value = minvalue(state.generateSuccessor(agentIndex, action), a, b, agentIndex + 1, depth)

                if value > v:
                    v = value
                    bestaction = action

                if v > b:
                    return v
                a = max(a, v)

            if depth == 0:  # If we're at the root level, return the best action
                return bestaction
            
            return v

        def minvalue(state, a, b, agentIndex, depth): # returns a utility value
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v = min(v, maxvalue(state.generateSuccessor(agentIndex, action), a, b, 0, depth + 1))
                else:
                    v = min(v, minvalue(state.generateSuccessor(agentIndex, action), a, b, agentIndex + 1, depth))

                if v < a:
                    return v
                b = min(b, v)
            return v

        agentIndex = 0 # Pacman plays first
        depth = 0

        return maxvalue(gameState, float('-inf'), float('inf'), agentIndex, depth)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def maxvalue(state, agentIndex, depth): # returns a utility value
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = float('-inf')
            for a in state.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v = max(v, chancevalue(state.generateSuccessor(agentIndex, a), 0, depth + 1))
                else:
                    v = max(v, chancevalue(state.generateSuccessor(agentIndex, a), agentIndex + 1, depth))
            
            return v

        def chancevalue(state, agentIndex, depth): # returns a utility value
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            sum = 0
            for a in state.getLegalActions(agentIndex):
                if len(state.getLegalActions(agentIndex)) != 0:
                    Pr = 1 / len(state.getLegalActions(agentIndex))
                else:
                    Pr = 0
                    
                if agentIndex == gameState.getNumAgents() - 1:
                    sum += Pr*maxvalue(state.generateSuccessor(agentIndex, a), 0, depth + 1)
                else:
                    sum += Pr*chancevalue(state.generateSuccessor(agentIndex, a), agentIndex + 1, depth)
            
            return sum
        
        agentIndex = 0 # Pacman plays first
        depth = 0

        bestscore = float('-inf')
        for a in gameState.getLegalActions(agentIndex):
            score = chancevalue(gameState.generateSuccessor(agentIndex, a), agentIndex + 1, depth)
            if score > bestscore:
                bestscore = score
                bestaction = a

        return bestaction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: All we have to do is reward PacMan more when he is close to food. In our initial
                 evaluate function, we had the same weight for food and ghosts. Here, we value food more.
                 Through testing, it seemed that only values in [8, 10] gave 6/6. This is a good balance
                 so that we don't underestimate ghosts, but also value food a little more.
    """
    pacPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Take into consideration the successorGameState.getScore()
    score = currentGameState.getScore()

    # The closer Pacman is to food, the better the state should be

    fooddists = []
    for food in foodPos.asList():
        fooddists.append(manhattanDistance(pacPos, food))

    distance_to_nearest_food = min(fooddists) if fooddists else 0
    
    # food proximity is more important!
    if distance_to_nearest_food != 0:
        score += 10 / distance_to_nearest_food
    else:
        score += 1000

    # If a ghost is close to Pacman, the state should be worse
    # Unless it is scared, where Pacman could move towards it
        
    ghostdists = []
    scaredghostdists = []
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        if ghostState.scaredTimer == 0:
            ghostdists.append(manhattanDistance(pacPos, ghostPos))
        else:
            scaredghostdists.append(manhattanDistance(pacPos, ghostPos))

    distance_from_ghost = min(ghostdists) if ghostdists else 0
    distance_from_scared_ghost = min(scaredghostdists) if scaredghostdists else 0

    if distance_from_ghost != 0:
        score -= 1 / distance_from_ghost
    else:
        score -= 1000
    score += distance_from_scared_ghost
        
    return score

# Abbreviation
better = betterEvaluationFunction
