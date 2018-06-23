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
import sys
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
#        print "scores", scores
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        "*** YOUR CODE HERE ***"
#        print successorGameState
#        print "newPos", newPos
#        print "newFood", newFood
        score = successorGameState.getScore()
        foodDistance = []
        for food in newFood.asList():
            foodDistance.append(manhattanDistance(newPos, food))
#        print "food dist",foodDistance
        if len(foodDistance) >0:
            score = score + ( 10 / min(foodDistance))
                
        ghostDistance = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if ghostDistance > 0:
            score = score - ( 10 / ghostDistance)
#        print "newGhostStates", newGhostStates
        return score

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
    def max_value(self, gameState, depth):
        legalMoves = gameState.getLegalActions(0)
        #if terminal state
        if depth > self.depth or gameState.isWin() or gameState.isLose() or not legalMoves:
#            print "Terminal"
            return self.evaluationFunction(gameState)
        
        actioncost = []
        for action in legalMoves:
            succ = gameState.generateSuccessor(0, action)
            actioncost.append((self.min_value(succ, 1, depth), action))
        
        return max(actioncost)
    
    def min_value(self, gameState, index, depth):
        ghostact = gameState.getLegalActions(index)
        #if terminal state
        if gameState.isWin() or gameState.isLose() or not ghostact:
            return self.evaluationFunction(gameState)
        succ = [gameState.generateSuccessor(index, action) for action in ghostact]
#           print "succ", succ
        if index == gameState.getNumAgents() - 1:
                return min([self.max_value(s, depth + 1) for s in succ])
        else:
                return min([self.min_value(s, index + 1, depth) for s in succ])

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 1)[1]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value(self, gameState, depth, alpha, beta):
        legalMoves = gameState.getLegalActions(0)
        
        if depth > self.depth or gameState.isWin() or gameState.isLose() or not legalMoves:
            return self.evaluationFunction(gameState), Directions.STOP
    
        v = -sys.maxint
        act = Directions.STOP
        for action in legalMoves:
            succ = gameState.generateSuccessor(0, action)
#            print "succ", succ
            cost = self.min_value(succ, 1, depth, alpha, beta)[0]
#            print "cost", cost
#            print "v", v
            if cost > v:
                v = cost
                act = action
            if v > beta:
                return v, act
            alpha = max(alpha, v)

        return v, act
    
    def min_value(self, gameState, index, depth, alpha, beta):
        ghostact = gameState.getLegalActions(index)
        
        if not ghostact or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
        v = sys.maxint
        act = Directions.STOP
    
        for action in ghostact:
            succ = gameState.generateSuccessor(index, action)
            if index == gameState.getNumAgents() - 1:
                cost = self.max_value(succ, depth + 1, alpha, beta)[0]
            else:
                cost = self.min_value(succ, index + 1, depth, alpha, beta)[0]
            
            
#            print "cost", cost
#            print "v", v

            if v > cost:
                v = cost
                act = action
            if alpha > v:
                return v, act
            beta = min(beta, v)

        return v, act

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 1, -sys.maxint, sys.maxint)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_value(self, gameState, depth):
        legalMoves = gameState.getLegalActions(0)
        #if terminal state
        if depth > self.depth or gameState.isWin() or gameState.isLose() or not legalMoves:
#            print "Terminal"
            return self.evaluationFunction(gameState), None
        
        actioncost = []
        for action in legalMoves:
            succ = gameState.generateSuccessor(0, action)
            actioncost.append((self.expected_value(succ, 1, depth)[0], action))
#        print "actcost",actioncost
        return max(actioncost)

    def expected_value(self, gameState, index, depth):
        ghostact = gameState.getLegalActions(index)
        
        if not ghostact or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None
    
        succs = [gameState.generateSuccessor(index, action) for action in ghostact]
#           print "succs", succs


        actioncost = []
        for succ in succs:
            if index == gameState.getNumAgents() - 1:
                actioncost.append(self.max_value(succ, depth + 1))
            else:
                actioncost.append(self.expected_value(succ, index + 1, depth))
        sum = 0
        for a in actioncost:
            sum = sum + float(a[0])
        averageScore = sum / len(actioncost)
#        print "AVG SCORE",averageScore
        return averageScore, None

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 1)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      Alter the value of score based on distances to ghost and food and the distance to the edible ghost
      calculate the score of the next state by even considering the ghost which is edible
      If the ghost is edible, move towards the ghost
      If it is a normal ghost, move away from it
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
#    print successorGameState
#    print "newPos", newPos
#    print "newFood", newFood

    score = currentGameState.getScore()
    
    # distance to ghosts
    ghostdist = 0
    for ghost in newGhostStates:
        dist = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if dist > 0:
            if ghost.scaredTimer > 0:
                score = score + 10 / dist
            else:
                score = score - 10 / dist

# distance to food
    foodDistance = []
    for food in newFood.asList():
        foodDistance.append(manhattanDistance(newPos, food))
#        print "food dist",foodDistance
    if len(foodDistance) >0:
        score = score + ( 10 / min(foodDistance))

    return score

# Abbreviation
better = betterEvaluationFunction

