import copy
import logging

__all__ = ['LogColorScheme', 'ColoredFormatter', 'DefaultLogFormat']

import typing


class LogColorScheme(object):
    def __init__(self, critical=None, error=None, warning=None, info=None, debug=None, time=None, name=None,
                 process=None, thread=None, lineno=None, funcname=None, filename=None, pathname=None):
        from pytools.pyutils.io.cli import CLIColorFormat

        self.critical = critical or CLIColorFormat('white', 'on_red')
        self.error = error or CLIColorFormat('red')
        self.warning = warning or CLIColorFormat('yellow')
        self.info = info or CLIColorFormat('blue')
        self.debug = debug or CLIColorFormat('grey')
        self.time = time or CLIColorFormat('green')
        self.name = name or CLIColorFormat('magenta')
        self.process = process or CLIColorFormat('blue')
        self.thread = thread or CLIColorFormat('yellow')
        self.lineno = lineno or CLIColorFormat('yellow', use_underline=True)
        self.funcname = funcname or CLIColorFormat('cyan')
        self.filename = filename or CLIColorFormat()
        self.pathname = pathname or CLIColorFormat('white', use_underline=True)


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', color_scheme: LogColorScheme = None):
        super().__init__(fmt, datefmt, style)
        self._scheme = color_scheme or LogColorScheme()

    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        record.levelname = getattr(self._scheme, record.levelname.lower()).colored(record.levelname)
        record.levelno = getattr(self._scheme, logging.getLevelName(record.levelno).lower()).colored(record.levelno)
        record.name = self._scheme.name.colored(record.name)
        record.process = self._scheme.process.colored(record.process)
        record.processName = self._scheme.process.colored(record.processName)
        record.thread = self._scheme.thread.colored(record.thread)
        record.threadName = self._scheme.thread.colored(record.threadName)
        record.lineno = self._scheme.lineno.colored(record.lineno)
        record.pathname = self._scheme.pathname.colored(record.pathname)
        record.funcName = self._scheme.funcname.colored(record.funcName)
        record.filename = self._scheme.filename.colored(record.filename)
        return super(ColoredFormatter, self).format(record)


class DefaultLogFormat(object):
    def __init__(self, use_time=True, use_time_msec=False, use_name=True, use_funcname=False, use_filename=False,
                 use_pathname=False, use_lineno=False, use_module=False, use_process=False, use_thread=False,
                 use_level=True, datefmt="%Y-%m-%d %H:%M:%S"):
        self.use_name = use_name
        self.use_funcname = use_funcname
        self.use_process = use_process
        self.use_thread = use_thread
        self.use_filename = use_filename
        self.use_pathname = use_pathname
        self.use_lineno = use_lineno
        self.use_module = use_module

        self.use_level = use_level
        self.datefmt = datefmt
        self.use_time = use_time
        self.use_time_msec = use_time_msec

    def build_context(self):
        context = []
        self.use_time and context.append('%(asctime)s.%(msecs)03d' if self.use_time_msec else '%(asctime)s')
        if self.use_lineno and self.use_pathname:
            context.append('%(pathname)s:%(lineno)s')
        else:
            self.use_pathname and context.append('%(pathname)s')
            self.use_filename and context.append('%(filename)s')
            self.use_lineno and context.append('%(lineno)s')
        self.use_name and context.append('%(name)s')
        self.use_funcname and context.append('%(funcName)s')
        self.use_process and context.append('%(processName)s')
        self.use_thread and context.append('%(threadName)s')
        self.use_module and context.append('%(module)s')
        return context

    def text_format(self):
        context = self.build_context()
        footer = '%(levelname)s' if self.use_level else ''

        caption = []
        if len(context) > 0:
            if len(context) == 1:
                caption.append(context[0])
            else:
                caption.append(' '.join((context[0], '(' + ' '.join(context[1:]) + ')')))
        footer and caption.append(footer)
        caption = ' '.join(caption)
        if caption:
            return f'{caption}: %(message)s'
        else:
            return f'%(message)s'

    def date_format(self):
        return self.datefmt

    def make_formatter(self, formatter_class, **kwargs):
        kwargs.setdefault('fmt', self.text_format())
        kwargs.setdefault('datefmt', self.date_format())
        return formatter_class(**kwargs)
