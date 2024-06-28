import os.path
import time

from commands.handle_numerical_command import *
from commands.load_raw_dataset_command import load_raw_dataset, load_raw_dataset_parquet
from commands.merge_dataset_header_command import merge_dataset_header
from commands.parse_header_description_command import parse_header_descriptions
from commands.parse_header_positions_command import parse_header_positions
from commands.remove_nan_on_target_command import remove_nan_on_target
from commands.set_binary_target_command import set_binary_target
from commands.split_train_analysis_command import split_train_analysis
from commands.split_train_test_command import split_train_test
from commands.store_train_test_analysis_command import store_train_test_analysis
from commands.unzip_raw_asc_file_command import unzip_raw_asc_file
from commands.write_dataset_as_parquet_command import write_dataset_as_parquet
from commands.write_headers_json_command import write_headers_json
from commands.set_types_command import set_types
from src.commands.select_filtered_headers_command import select_filtered_columns

initial_commands = [
    unzip_raw_asc_file,
    parse_header_descriptions,
    parse_header_positions,
    merge_dataset_header,
    write_headers_json,
    load_raw_dataset,
    write_dataset_as_parquet('../resources/processed/raw_dataset.parquet'),
]

try_to_load_commands = [
    parse_header_descriptions,
    parse_header_positions,
    merge_dataset_header,
    load_raw_dataset_parquet
]

if os.path.exists('../resources/processed/raw_dataset.parquet'):
    init_commands = try_to_load_commands
else:
    init_commands = initial_commands

commands = init_commands + [
    remove_nan_on_target,
    handle_numerical_categories,
    set_types,
    select_filtered_columns,
    set_binary_target,
    split_train_test,
    split_train_analysis,
    store_train_test_analysis,
]

if __name__ == '__main__':
    context = {}
    for command in commands:
        start = time.time()
        print('running {}'.format(command.__name__))
        result = command(context)
        end = time.time()
        print('{} ran in {} ms'.format(command.__name__, end - start))

        if not result:
            print('stopping at {}'.format(command.__name__))
            break
