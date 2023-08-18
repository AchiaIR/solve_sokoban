
from yacs.config import CfgNode as CN

############################################
# define default configuration for parsing #
############################################

defaultCFG = CN()

# Model Part
defaultCFG.ALGORITHM = CN()
defaultCFG.ALGORITHM.NAME = 'DeepQLearning'
defaultCFG.ALGORITHM.NUM_EPISODES = 500
defaultCFG.ALGORITHM.NUM_STEPS = 500
defaultCFG.ALGORITHM.DISCOUNT_FACTOR = 0.9
defaultCFG.ALGORITHM.EPSILON_GREEDY = 0.99
defaultCFG.ALGORITHM.DECAY = 0.99
defaultCFG.ALGORITHM.CHANGE_REWARD = False

# sokoban part
defaultCFG.SOKOBAN = CN()
defaultCFG.SOKOBAN.DIM = 7
defaultCFG.SOKOBAN.NUM_BOXES = 1
defaultCFG.SOKOBAN.FIX = False