import sys


def print_none(x):
    pass


def no_output(func):
    def wrapper(*args, **kwargs):
        sys.stdout = None
        result = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return result
    return wrapper
