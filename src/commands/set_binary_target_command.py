

def set_binary_target(context: dict) -> bool:
    df = context['raw_dataset']
    target_class = 1
    df['target'] = (df['CVDINFR4'] == target_class).astype(int)
    context['raw_dataset'] = df
    return True
