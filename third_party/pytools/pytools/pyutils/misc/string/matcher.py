import fnmatch
from typing import List, Union

__all__ = ['WildCardMatcher']


class WildCardMatcher(object):
    def __init__(self, patterns: Union[str, List[str]]):
        if isinstance(patterns, str):
            patterns = [patterns]
        self._patterns = patterns

    def match(self, text: str):
        for pattern in self._patterns:
            if fnmatch.fnmatch(text, pattern):
                return True
        return False
