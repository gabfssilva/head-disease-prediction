import re

def merge_dataset_header(context: dict) -> bool:
    positions = context['header_positions']
    descriptions = context['header_descriptions']

    merged = {**descriptions}

    for key in descriptions.keys():
        if key in positions:
            label = descriptions[key]['label']
            normalized_key = label.lower().replace(' ', '_')
            normalized_key = re.sub(r'[^\w\s]', '', normalized_key)

            merged[key]['starts_at'] = positions[key]['from']
            merged[key]['ends_at'] = positions[key]['to']
            merged[key]['normalized_key'] = normalized_key
        else:
            del merged[key]

    context['headers'] = merged
    return True
