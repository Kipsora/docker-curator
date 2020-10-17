import functools
import inspect
import warnings

__all__ = ["check_fn", "synchronized_member_fn", "deprecated", "cached_property", "classproperty",
           'check_failed_message']


def check_fn(checker, message=None):
    if inspect.signature(checker).parameters:
        raise RuntimeError("Check decorator must have zero parameters")

    def wrapper(fn=None):
        if callable(fn):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                if not checker():
                    nonlocal message
                    message = message or getattr(checker, "__failed_message__", f"Cannot pass the checker: {checker}")
                    raise RuntimeError(message)
                return fn(*args, **kwargs)

            return wrapped
        else:
            checker()

    return wrapper


def check_failed_message(message):
    def wrapper(fn):
        fn.__failed_message__ = message
        return fn

    return wrapper


def synchronized_member_fn(fn_or_lock_member):
    if callable(fn_or_lock_member):
        return synchronized_member_fn('_lock')(fn_or_lock_member)

    def wrapper(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            lock = getattr(self, fn_or_lock_member)
            lock.acquire()
            try:
                return fn(self, *args, **kwargs)
            finally:
                lock.release()

        return wrapped

    return wrapper


def deprecated(fn_or_docs):
    def wrapper(fn):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used."""

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if not getattr(fn, "__deprecated__", False):
                warnings.simplefilter('always', DeprecationWarning)  # turn off filter
                warnings.warn(f"Call to a deprecated function {fn.__name__}." if not fn_or_docs else str(fn_or_docs),
                              category=DeprecationWarning,
                              stacklevel=2)
                warnings.simplefilter('default', DeprecationWarning)  # reset filter
            setattr(fn, "__deprecated__", True)
            return fn(*args, **kwargs)

        return wrapped

    if callable(fn_or_docs):
        return deprecated(None)(fn_or_docs)
    else:
        return wrapper


class cached_property(property):
    """Descriptor that mimics @property but caches output in member variable."""

    def __get__(self, obj, objtype=None):
        # See https://docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:  # pytype: disable=attribute-error
            raise AttributeError('unreadable attribute')
        attr = '__cached_' + self.fget.__name__  # pytype: disable=attribute-error
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)  # pytype: disable=attribute-error
            setattr(obj, attr, cached)
        return cached


class classproperty(property):
    """Descriptor to be used as decorator for @classmethods."""

    def __get__(self, obj, objtype=None):
        return self.fget.__get__(None, objtype)()  # pytype: disable=attribute-error
