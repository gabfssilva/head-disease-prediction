

class Command:
    def handle(self, context: dict) -> bool:
        raise NotImplementedError("Implement this method")

    def name(self) -> str:
        return self.__class__.__name__
