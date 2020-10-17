import abc
import math

__all__ = ['NumberFormatter', 'TemplateFormatter', 'SecondsFormatter']


class NumberFormatter(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, value):
        raise NotImplementedError


class TemplateFormatter(NumberFormatter):
    def __init__(self, template: str):
        self._template = template

    def __call__(self, value):
        return self._template.format(value)


class SecondsFormatter(NumberFormatter):
    MINUTE = 60
    HOUR = 60 * MINUTE
    DAY = 24 * HOUR

    def __call__(self, value):
        result = []
        value = float(value)

        if value >= self.DAY * 2:
            num_days = math.floor(value / self.DAY)
            value -= num_days * self.DAY
            result.append(f'{num_days}d')

        if value >= self.HOUR * 3:
            num_hours = math.floor(value / self.HOUR)
            value -= num_hours * self.HOUR
            result.append(f'{num_hours}h')

        if value >= self.MINUTE:
            num_minutes = math.floor(value / self.MINUTE)
            value -= num_minutes * self.MINUTE
            result.append(f'{num_minutes}m')

        if value > 0:
            result.append(f'{value:.2f}s')

        return ' '.join(result)
