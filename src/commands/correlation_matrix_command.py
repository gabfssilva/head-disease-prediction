from src.commands.command import Command

import plotly.express as px

def correlation_matrix(context: dict) -> bool:
    dataset = context['analysis_dataframe']
    correlation_matrix = dataset.corr(method='spearman')

    fig = px.imshow(correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    labels={'x': "Feature", 'y': "Feature", 'color': "Correlation Coefficient"},
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns)

    fig.update_layout(title="BRFSS Correlation Matrix",
                      xaxis_title="Features",
                      yaxis_title="Features")

    fig.show()

    return True
