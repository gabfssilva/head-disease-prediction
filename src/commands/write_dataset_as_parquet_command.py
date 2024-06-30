from typing import Callable

def write_dataset_as_parquet(filepath: str, dataset: str = 'raw_dataset') -> Callable[[dict], bool]:
    def with_context(context) -> bool:
        df = context[dataset]
        df.to_parquet(filepath, index=False)
        return True

    return with_context
