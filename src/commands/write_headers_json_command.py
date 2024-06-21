import json


def write_headers_json(context: dict) -> bool:
    with open('resources/processed/headers.json', 'w') as file:
        json.dump(context['headers'], file)

    return True
