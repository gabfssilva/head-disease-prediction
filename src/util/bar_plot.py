import pandas as pd
import plotly.express as px

def plot_data(df, col, headers, column_name_mapper = lambda x: x):
    target = 'CVDSTRK3'

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

    if column_name_mapper is not None:
        df[col] = df[col].map(lambda x: column_name_mapper(safe_map(x, col_info)))

    df_count = df.groupby([target, col]).size().reset_index(name='count')

    df_count = df_count.sort_values('count', ascending=False)

    total_count = df_count['count'].sum()

    fig = px.bar(df_count, x=col, y='count', color=target,
                 title=f"Total Counts by Category - Total: {total_count}",
                 labels={
                     target: target_info['question'],
                     col: headers[col]['question'],
                     'count': 'Count'
                 })

    fig.update_layout(autosize=True, xaxis_title=headers[col]['question'], yaxis_title="Count")

    fig.show()
