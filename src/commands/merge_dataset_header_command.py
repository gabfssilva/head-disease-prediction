from src.commands.command import Command


class MergeDatasetHeaderCommand(Command):
    def handle(self, context: dict) -> bool:
        positions = context['header_positions']
        descriptions = context['header_descriptions']

        merged = {**descriptions}

        for key in descriptions.keys():
            if key in positions:
                merged[key]['starts_at'] = positions[key]['from']
                merged[key]['ends_at'] = positions[key]['to']
            else:
                del merged[key]

        context['headers'] = merged
        return True
