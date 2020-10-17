__all__ = ["set_python_seed"]


def set_python_seed(seed):
    import random
    random.seed(seed)
