import logging

__all__ = ['CallbackHandler', 'make_handler']

from pytools.pyutils.logging.formatter import DefaultLogFormat


class CallbackHandler(logging.Handler):
    def __init__(self, callback, level=logging.NOTSET):
        super().__init__(level)
        assert callable(callback)
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._callback(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def make_handler(handler_class, level='NOTSET',
                 format_kwargs=None,
                 formatter_class=logging.Formatter,
                 formatter_kwargs=None,
                 handler_kwargs=None,
                 log_format=DefaultLogFormat) -> logging.Handler:
    log_format = log_format(**(format_kwargs or dict()))
    formatter = log_format.make_formatter(formatter_class, **(formatter_kwargs or dict()))

    assert issubclass(handler_class, logging.Handler)
    handler = handler_class(**(handler_kwargs or dict()))
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler
