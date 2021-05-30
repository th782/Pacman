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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE Q1 ***"
        food_list = childGameState.getFood().asList()
        Food = float("inf")
        for food in food_list:
            Food = min(Food, manhattanDistance(newPos, food))

        for ghost_pos in childGameState.getGhostPositions():
            if manhattanDistance(newPos, ghost_pos) < 2:
                return -float('inf')
        return childGameState.getScore() + (1.0 / Food)

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """ 
        "*** YOUR CODE HERE Q2 ***"
        return self.max_value(gameState, 0, 0)[0]
        
    def min_value(self, gameState, agentIndex, depth):
        best_action = ("min", float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            success_action = (action, self.min_max(gameState.getNextState(agentIndex, action),
                                                   (depth + 1) % gameState.getNumAgents(), depth + 1))
            best_action = min(best_action, success_action, key=lambda x: x[1])
        return best_action

    def min_max(self, gameState, agentIndex, depth):
        if depth == self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth)[1]
        else:
            return self.min_value(gameState, agentIndex, depth)[1]

    def max_value(self, gameState, agentIndex, depth):
        best_action = ("max", -float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            success_action = (action, self.min_max(gameState.getNextState(agentIndex, action),
                                                   (depth + 1) % gameState.getNumAgents(), depth + 1))
            best_action = max(best_action, success_action, key=lambda x: x[1])
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE Q3 ***"
        return self.max_value(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        best_action = ("min", float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            success_action = (action, self.alpha_beta(gameState.getNextState(agentIndex, action),
                                                      (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            best_action = min(best_action, success_action, key=lambda x: x[1])
            if best_action[1] < alpha:
                return best_action
            else:
                beta = min(beta, best_action[1])
        return best_action

    def alpha_beta(self, gameState, agentIndex, depth, alpha, beta):
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            return self.max_value(gameState, agentIndex, depth, alpha, beta)[1]
        else:
            return self.min_value(gameState, agentIndex, depth, alpha, beta)[1]

    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        best_action = ("max", -float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            success_action = (action, self.alpha_beta(gameState.getNextState(agentIndex, action),
                                                      (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            best_action = max(best_action, success_action, key=lambda x: x[1])
            if best_action[1] > beta:
                return best_action
            else:
                alpha = max(alpha, best_action[1])
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE Q4 ***"
        max_depth = self.depth * gameState.getNumAgents()
        return self.expect_i_max(gameState, "expect", max_depth, 0)[0]

    def max_value(self, gameState, action, depth, agentIndex):
        best_action = ("max", -(float('inf')))
        legal_actions = gameState.getLegalActions(agentIndex)
        for legal_action in legal_actions:
            next_agent = (agentIndex + 1) % gameState.getNumAgents()
            success_action = None
            if depth != self.depth * gameState.getNumAgents():
                success_action = action
            else:
                success_action = legal_action
            success_value = self.expect_i_max(gameState.getNextState(agentIndex, legal_action),
                                              success_action, depth - 1, next_agent)
            best_action = max(best_action, success_value, key=lambda x: x[1])
        return best_action

    def expect_value(self, gameState, action, depth, agentIndex):
        legal_actions = gameState.getLegalActions(agentIndex)
        average_score = 0
        prop = 1.0 / len(legal_actions)
        for legal_action in legal_actions:
            next_agent = (agentIndex + 1) % gameState.getNumAgents()
            best_action = self.expect_i_max(gameState.getNextState(agentIndex, legal_action),
                                            action, depth - 1, next_agent)
            average_score += best_action[1] * prop
        return action, average_score

    def expect_i_max(self, gameState, action, depth, agentIndex):

        if depth == 0 or gameState.isLose() or gameState.isWin():
            return action, self.evaluationFunction(gameState)
        if agentIndex is 0:
            return self.max_value(gameState, action, depth, agentIndex)
        else:
            return self.expect_value(gameState, action, depth, agentIndex)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE Q5 ***"
    
    food_left = currentGameState.getNumFood()
    capsules_left = len(currentGameState.getCapsules())

    food_left_M = 950000
    capsules_left_M = 10000
    food_distance_M = 1000

    added_factors = 0
    if currentGameState.isLose():
        added_factors -= 50000
    elif currentGameState.isWin():
        added_factors += 50000

    new_position = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()

    Food = float('inf')
    for food in food_list:
        Food = min(Food, manhattanDistance(new_position, food))

    ghost_distance = 0
    for ghost in currentGameState.getGhostPositions():
        ghost_distance = manhattanDistance(new_position, ghost)
        if ghost_distance < 2:
            return -float('inf')

    eval = 1.0 / (food_left + 1) * food_left_M + ghost_distance
    eval += 1.0 / (Food + 1) * food_distance_M
    eval += 1.0 / (capsules_left + 1) * capsules_left_M + added_factors

    return eval

# Abbreviation
better = betterEvaluationFunction
