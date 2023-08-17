import numpy as np


class Distribution:

    def sample(self):
        '''
        Returns a numpy array.
        '''
        raise NotImplementedError


class FiniteDistribution(Distribution):
    '''
    Uniform distribution that selects from a finite list.

    Parameters:
        list_of_points: list
    '''

    def __init__(self, list_of_points=[]):
        self.points = list_of_points
        self.num_points = len(self.points)

    def sample(self):
        index = np.random.randint(self.num_points)
        return self.points[index]

    def add_points(self, list_of_points):
        self.points = self.points + list_of_points
        self.num_points = len(self.points)


class ProductDist(Distribution):
    '''
    Product distribution for combined system and resource states.
    '''

    def __init__(self, sys_dist, res_model):
        self.sys_dist = sys_dist
        self.res_model = res_model

    def sample(self):
        if self.res_model is None:
            res_state = np.array([])
        else:
            res_state = self.res_model.res_init
        return np.concatenate([self.sys_dist.sample(), res_state])
