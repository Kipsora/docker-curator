import warnings

__all__ = ['CLIColorFormat']


class CLIColorFormat(object):
    def __init__(self,
                 color=None,
                 background=None,
                 use_bold=False,
                 use_dark=False,
                 use_blink=False,
                 use_underline=False,
                 use_reverse=False,
                 use_concealed=False):
        self._color = color
        self._background = background
        self._attributes = {
            'bold': use_bold,
            'dark': use_dark,
            'blink': use_blink,
            'reverse': use_reverse,
            'underline': use_underline,
            'concealed': use_concealed
        }

    @property
    def color(self):
        return self._color

    @property
    def background(self):
        return self._background

    @property
    def attributes(self):
        return [k for k, v in self._attributes.items() if v]

    def set_attribute(self, attribute, inuse=True):
        assert attribute in self._attributes
        self._attributes[attribute] = inuse
        return self

    def set_background(self, background):
        self._background = background
        return self

    def set_color(self, color):
        self._color = color
        return self

    def colored(self, text):
        try:
            import termcolor
            return termcolor.colored(text, self._color, self._background, self.attributes)
        except ModuleNotFoundError:
            warnings.warn('Unable to import termcolor package.')
            return text
