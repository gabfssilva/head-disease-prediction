import zipfile

from src.commands.command import Command


class UnzipRawASCFileCommand(Command):
    def handle(self, context: dict) -> bool:
        zip_name = '../resources/dataset.zip'

        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall('../resources/generated/dataset')

        return True
