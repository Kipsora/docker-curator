import datetime

__all__ = ['current_time']


def current_time(fmt='%Y-%m-%d %H:%M:%S'):
    return datetime.datetime.now().strftime(fmt)
