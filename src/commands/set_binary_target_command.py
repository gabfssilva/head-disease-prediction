

def set_binary_target(context: dict) -> bool:
    df = context['raw_dataset']
    df['target'] = (df['CVDINFR4'] == '1').astype(str)
    context['raw_dataset'] = df
    return True
