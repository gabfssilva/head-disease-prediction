
def handle_numerical_categories(context: dict) -> bool:
    dataset = context['raw_dataset']
    headers_df = context['headers']

    numerical = []
    for k, v in headers_df.items():
        for a in v['possible_answers'].keys():
            if ' - ' in a:
                numerical.append(k)

    for col in dataset.columns:
        if col not in numerical:
            dataset[col] = dataset[col].astype('category')

    invalid_values = {
        'HHADULT': [88, 77, 99],
        'PHYSHLTH': [88, 77, 99],
        'MENTHLTH': [88, 77, 99],
        'POORHLTH': [88, 77, 99],
        'SLEPTIM1': [77, 99],
        'CHILDREN': [88, 99],
        'WEIGHT2': [7777, 9999],
        'HEIGHT3': [7777, 9999],
        'LCSFIRST': [777, 888, 999],
        'LCSLAST': [777, 999],
        'LCSNUMCG': [777, 999],
        'ALCDAY4': [777, 888, 999],
        'AVEDRNK3': [88, 77, 99],
        'DRNK3GE5': [88, 77, 99],
        'MAXDRNKS': [77, 99],
        'FLSHTMY3': [777777, 999999],
        'HIVTSTD3': [777777, 999999],
        'CHKHEMO3': [88, 98, 77, 99],
        'HPVADSHT': [77, 99],
        'COVIDFS1': [777777, 999999],
        'COVIDSE1': [777777, 999999],
        'COPDSMOK': [88, 77, 99],
        'CNCRAGE': [98, 99],
        'MARIJAN1': [88, 77, 99],
        'DROCDY4_': [900],
        '_DRNKWK2': [99900]
    }

    def replace_with_none(df, column, invalid_list):
        df[column] = df[column].replace(invalid_list, None)
        return df

    for column, invalids in invalid_values.items():
        dataset = replace_with_none(dataset, column, invalids)

    context['raw_dataset'] = dataset

    return True
