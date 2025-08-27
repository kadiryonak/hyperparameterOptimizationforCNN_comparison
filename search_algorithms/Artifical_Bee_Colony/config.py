import numpy as np

class Config:
    def __init__(self):

        self.NUMBER_OF_POPULATION = 20
        self.MAXIMUM_EVALUATION = 40
        self.LIMIT = 10
        self.FOOD_NUMBER = int(self.NUMBER_OF_POPULATION / 2)

        self.DIMENSION = 6

        self.LOWER_BOUND = np.array([8, 1e-5, 8, 8, 32, 0.0])
        self.UPPER_BOUND = np.array([128, 1e-2, 128, 128, 512, 0.5])

        self.MINIMIZE = True
        self.SHOW_PROGRESS = True
        self.RANDOM_SEED = False
        self.SEED = 42


