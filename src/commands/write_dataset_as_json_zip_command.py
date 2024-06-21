import io
import zipfile


def write_dataset_as_json_zip(self, context: dict) -> bool:
    buffer = io.StringIO()
    context['raw_dataset'].to_json(path_or_buf=buffer, orient='records')

    path = 'resources/generated/raw_dataset.json.zip'
    filename = 'raw_dataset.json'
    content = buffer.getvalue()

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as ref:
        ref.writestr(filename, data=content)
        ref.testzip()

    return True
