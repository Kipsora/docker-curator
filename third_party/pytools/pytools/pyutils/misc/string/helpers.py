import typing

__all__ = ['remove_suffix', 'remove_prefix', 'row_pad_prefix']


def remove_suffix(string: str, suffix: str):
    if string.endswith(suffix):
        return string[:len(string) - len(suffix)]
    return string


def remove_prefix(string: str, prefix: str):
    if string.startswith(prefix):
        return string[len(prefix):]
    return string


def row_pad_prefix(strings: str or typing.List[str], prefix: str):
    if isinstance(strings, str):
        strings = strings.split('\n')
    return '\n'.join([prefix + string for string in strings])
