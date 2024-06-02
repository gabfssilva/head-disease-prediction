from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

def write_dataset_as_parquet(filepath: str) -> Callable[[dict], bool]:
    def with_context(context) -> bool:
        table = pa.Table.from_pandas(context['raw_dataset'])
        headers = context['headers']

        fields = []
        for field in table.schema:
            header = headers[field.name]
            label = header['label']

            metadata = {
                'label': label,
                'question': header['question'],
                'possible_answers': str(header['possible_answers'])
            }

            new_field = pa.field(field.name, field.type, metadata=metadata)
            fields.append(new_field)

        new_schema = pa.schema(fields)
        new_table = pa.Table.from_arrays(arrays=table.columns, schema=new_schema)
        pq.write_table(new_table, filepath)

        return True

    return with_context
