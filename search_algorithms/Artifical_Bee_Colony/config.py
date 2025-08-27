import numpy as np

class Config:
    def __init__(self):
        # colony size (total bees)
        self.NUMBER_OF_POPULATION = 20
        self.MAXIMUM_EVALUATION = 40
        self.LIMIT = 10
        self.FOOD_NUMBER = int(self.NUMBER_OF_POPULATION / 2)
        # vector dimension used by ABC (for CNN we use 6 dims)
        self.DIMENSION = 6
        # bounds order: [batch_size, lr, conv1, conv2, fc1, dropout]
        self.LOWER_BOUND = np.array([8, 1e-5, 8, 8, 32, 0.0])
        self.UPPER_BOUND = np.array([128, 1e-2, 128, 128, 512, 0.5])
        # minimization flag
        self.MINIMIZE = True
        self.SHOW_PROGRESS = True
        # deterministic/seeding
        # NOTE: RANDOM_SEED == False => we set the seed (deterministic). Set to True if you WANT randomness.
        self.RANDOM_SEED = False
        self.SEED = 42

