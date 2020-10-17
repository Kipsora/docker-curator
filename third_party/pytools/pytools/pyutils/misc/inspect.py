import inspect

__all__ = ['get_callee_module', 'get_caller_module']


def get_caller_module():
    frame_info = inspect.stack()[2]
    module = inspect.getmodule(frame_info[0])
    return module


def get_callee_module():
    frame_info = inspect.stack()[1]
    module = inspect.getmodule(frame_info[0])
    return module
