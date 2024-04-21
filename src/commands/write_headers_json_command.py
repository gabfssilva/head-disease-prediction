import json

from src.commands.command import Command


class WriteHeadersJsonCommand(Command):
    def handle(self, context: dict) -> bool:
        with open('../resources/generated/headers.json', 'w') as file:
            json.dump(context['headers'], file)

        return True
