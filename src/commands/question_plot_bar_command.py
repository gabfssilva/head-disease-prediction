from typing import Callable

from src.commands.command import Command

import plotly.io as pio
import plotly.graph_objects as go


def question_plot_bar(feature: str) -> Callable[[dict], bool]:
    pio.renderers.default = ''

    def question_plot_bar_f(context: dict) -> bool:
        headers_df = context['headers']
        dataset = context['analysis_dataframe']

        def possible_answers(question_key: str) -> dict:
            return {
                int(k): v for k, v in headers_df[question_key]['possible_answers'].items() if k.isdigit()
            }

        def count_of(question_key: str):
            return dataset[question_key].map(possible_answers(question_key)).value_counts()

        def plot_bar_of_count(question_key: str):
            value_counts = count_of(question_key)

            fig = go.Figure(
                data=[go.Bar(x=value_counts.index, y=value_counts.values)]
            )

            fig.update_layout(
                title=headers_df[question_key]['label'],
                xaxis_title='Categories',
                yaxis_title='Count',
                width=1100,
                height=600,
            )

            fig.show()

        return plot_bar_of_count(feature)

    return question_plot_bar_f
