import os.path
import time

from src.commands import unzip_raw_asc_file_command, parse_header_positions_command, merge_dataset_header_command, \
    load_raw_dataset_command
from src.commands.describe_analysis_dataframe_command import describe
from src.commands.handle_numerical_command import *
from src.commands.load_raw_dataset_command import load_raw_dataset, load_raw_dataset_parquet
from src.commands.merge_dataset_header_command import merge_dataset_header
from src.commands.parse_header_description_command import parse_header_descriptions
from src.commands.parse_header_positions_command import parse_header_positions
from src.commands.question_plot_bar_command import question_plot_bar
from src.commands.remove_nan_on_target_command import remove_nan_on_target
from src.commands.split_train_analysis_command import split_train_analysis
from src.commands.split_train_test_command import split_train_test
from src.commands.unzip_raw_asc_file_command import unzip_raw_asc_file
from src.commands.write_dataset_as_parquet_command import write_dataset_as_parquet
from src.commands.write_headers_json_command import write_headers_json

initial_commands = [
    unzip_raw_asc_file,
    parse_header_descriptions,
    parse_header_positions,
    merge_dataset_header,
    write_headers_json,
    load_raw_dataset,
    write_dataset_as_parquet('../resources/generated/raw_dataset.parquet'),
]

try_to_load_commands = [
    parse_header_descriptions,
    parse_header_positions,
    merge_dataset_header,
    load_raw_dataset_parquet
]

if os.path.exists('../resources/generated/raw_dataset.parquet'):
    init_commands = try_to_load_commands
else:
    init_commands = initial_commands

commands = init_commands + [
    remove_nan_on_target,
    handle_numerical_categories,
    split_train_test,
    split_train_analysis,
    # write_dataset_as_parquet('../resources/generated/df_analysis_10pct.parquet', 'analysis_dataframe'),
    # describe,
    lambda context: context['analysis_dataframe'].to_csv('../resources/generated/df_analysis_10pct.csv')
    # questions_plot_bar(
    #     ['_AGE_G', 'CVDSTRK3'],
    #     lambda field: field.split('Notes')[0]
    # ),
    # question_plot_bar(
    #     'WEIGHT2',
    #     lambda field: field
    # )
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
