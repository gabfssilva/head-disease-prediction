import pandas as pd
import plotly.express as px

def plot_data(
    df,
    col,
    headers,
    sort = "desc",
    target_cols = ["Don’t know/Not sure", 'No', 'Refused', 'Yes'],
    column_name_mapper = lambda x: x
):
    target = 'CVDINFR4'

    if target not in headers:
        raise ValueError(f"Header information for {target} not available in headers.")

    target_info = headers[target]
    col_info = headers[col]

    def safe_map(x, info):
        try:
            return info['possible_answers'][str(int(x))]
        except (ValueError, TypeError):
            return 'Unknown'

    df[target] = df[target].map(lambda x: safe_map(x, target_info))
    df = df[df[target].isin(target_cols)]

    if column_name_mapper is not None:
        df[col] = df[col].map(lambda x: insert_line_breaks(column_name_mapper(safe_map(x, col_info)), 12))

    df_count = df.groupby([target, col]).size().reset_index(name='count')
    total_count = df_count['count'].sum()
    df_count['percentage'] = (df_count['count'] / total_count) * 100

    wrapped_title = insert_line_breaks(headers[col]['question'])

    fig = px.bar(df_count, x=col, y='percentage', color=target,
                 title=f"Percentage by {col} - Total: {total_count}",
                 labels={
                     target: 'Myocardial Infarction',
                     # col: wrapped_title,
                     'percentage': 'Percentage (%)'
                 },
                 color_discrete_map={
                     'Yes': '#fbad26',
                     'No': '#0063a3',
                     'Don’t know/Not sure': '#6a6e79',
                     'Refused': '#1e8a44'
                 })

    if sort == None:
        xaxis = {}

        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ))
    elif type(sort) is list:
        xaxis = {
            'categoryorder': 'array',
            'categoryarray': list(map(lambda x: insert_line_breaks(x, 12), sort))
        }

        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ))
    else:
        xaxis = {'categoryorder': f'total {sort}ending'}

        if sort == 'asc':
            fig.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))
        else:
            fig.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ))

    # fig.update_layout(autosize=True, xaxis_title=wrapped_title, yaxis_title="Percentage (%)")
    fig.update_layout(autosize=True,
                      xaxis=xaxis,
                      yaxis_title="Percentage (%)")
    fig.write_image(f'../resources/generated/{target}_{col}_{'-'.join(target_cols).replace("'", "").replace("/", "")}.png', format='png', scale=4)
    fig.show()

def insert_line_breaks(title, max_len=60):
    words = title.split()
    final_title = ""
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > max_len:
            final_title += current_line + '<br>'
            current_line = word + ' '
        else:
            current_line += word + ' '
    final_title += current_line
    return final_title.strip()
