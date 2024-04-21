import pandas as pd

from src.commands.command import Command


class LoadRawDatasetCommand(Command):
    def handle(self, context: dict) -> bool:
        headers = context['headers']

        context['raw_dataset'] = pd.read_fwf(
            filepath_or_buffer='../resources/generated/dataset/LLCP2022.ASC ',
            colspecs=list(map(lambda x: (x['starts_at']-1, x['ends_at']-1), headers.values())),
            names=list(headers.keys()),
            header=None
        )

        return True
