__all__ = ['set_numpy_seed']


def set_numpy_seed(seed):
    import numpy
    numpy.random.seed(seed)
