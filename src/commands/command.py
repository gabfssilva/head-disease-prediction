from dataclasses import dataclass
from typing import Callable


class Command:
    def skip(self, context: dict) -> bool:
        return False

    def handle(self, context: dict) -> bool:
        raise NotImplementedError("Implement this method")

    def name(self) -> str:
        return self.__class__.__name__
