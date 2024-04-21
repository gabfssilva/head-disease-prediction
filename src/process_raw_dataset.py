import time

from src.commands.load_raw_dataset_command import LoadRawDatasetCommand
from src.commands.merge_dataset_header_command import MergeDatasetHeaderCommand
from src.commands.parse_header_description_command import ParseHeaderDescriptionsCommand
from src.commands.parse_header_positions_command import ParseHeaderPositionsCommand
from src.commands.unzip_raw_asc_file_command import UnzipRawASCFileCommand
from src.commands.write_dataset_as_json_zip_command import WriteDatasetAsJsonZipCommand
from src.commands.write_headers_json_command import WriteHeadersJsonCommand

commands = [
    UnzipRawASCFileCommand(),
    ParseHeaderDescriptionsCommand(),
    ParseHeaderPositionsCommand(),
    MergeDatasetHeaderCommand(),
    WriteHeadersJsonCommand(),
    LoadRawDatasetCommand(),
    WriteDatasetAsJsonZipCommand()
]

if __name__ == '__main__':
    context = {}
    for command in commands:
        start = time.time()
        print('running {}'.format(command.name()))
        result = command.handle(context)
        end = time.time()
        print('{} ran in {} ms'.format(command.name(), end - start))

        if not result:
            print('stopping at {}'.format(command.name()))
            break
