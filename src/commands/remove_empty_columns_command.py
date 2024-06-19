
def remove_empty_columns(context: dict) -> bool:
    df = context['raw_dataset']
    df = df.drop(['TOLDCFS', 'HAVECFS', 'WORKCFS', 'COLGHOUS', 'COLGSEX1'], axis=1)
    context['raw_dataset'] = df
    return True
