from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, precision_recall_curve
from sk.enhanced_grid_search import EnhancedGridSearchCV
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

strongcyan = '#12AACC'
gold = 'gold'
orange = 'orange'

colors = ['mediumturquoise', strongcyan, gold, orange, '#007EB0']

def plot_confusion_matrix(
    estimator: EnhancedGridSearchCV, 
    X_test=pd.DataFrame({}), 
    y_test=pd.Series([])
) -> None:
    classes = estimator.classes_
    conf_matrix = estimator.result_.best.confusion_matrix

    fig = px.imshow(conf_matrix,
        labels=dict(
          x="Predicted", 
          y="Actual", 
        ),
        text_auto=True,
        x=classes,
        y=classes,
        color_continuous_scale=[strongcyan, orange, gold]
    )

    fig.show()

    if not X_test.empty and not y_test.empty:
        y_pred = estimator.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)

        fig = px.imshow(conf_matrix,
            labels=dict(
                x="Predicted", 
                y="Actual", 
            ),
            text_auto=True,
            x=classes,
            y=classes,
            color_continuous_scale=[strongcyan, orange, gold]
        )

        fig.show()


def distribution_plot(estimator):
    best_scores = estimator.result_.best.scores.items()
    num_plots = len(best_scores)
    fig = go.Figure()

    for index, (name, scores) in enumerate(best_scores):
        fig.add_trace(go.Histogram(
            x=scores['validation'], 
            nbinsx=20, 
            name=f'{name} Distribution',
            marker_color='blue',
            opacity=0.7
        ))

    fig.update_layout(
        title='Score Distributions',
        barmode='overlay',
        xaxis_title='Score',
        yaxis_title='Density',
        width=900,
        height=300 * num_plots,
    )

    fig.update_traces(histnorm='probability')

    fig.show()

def precision_recall_curve_plot(estimator):
    recall = estimator.result_.best.scores['recall']['validation']
    precision = estimator.result_.best.scores['precision']['validation']

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=recall, y=precision, mode='markers+lines',
                             name='Precision vs. Recall', marker=dict(symbol='circle')))

    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(scaleanchor="x", scaleratio=1, range=[0, 1.0]),
        yaxis=dict(scaleanchor="y", scaleratio=1, range=[0, 1.05])
    )

    fig.show()

def roc_curve_plot(estimator):
    classes = estimator.classes_
    n_classes = len(classes)
    probabilities = estimator.result_.best.predicted_probabilities
    probability_array = np.array([[prob[label] for label in classes] for prob in probabilities])

    y_true = estimator.result_.fitted_with.y

    if n_classes > 2:
        y_true = label_binarize(y_true, classes=np.unique(y_true))

    fig = go.Figure()

    for i in range(n_classes):
        if n_classes == 2 and i == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i] if n_classes > 2 else y_true, probability_array[:, i], pos_label=estimator.positive_class)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f'ROC curve of class {i} (area = {roc_auc:.2f})',
                                 line=dict(width=2)))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             line=dict(color='navy', width=2, dash='dash'),
                             name='Chance'))

    fig.update_layout(
        title='ROC Curve(s)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(scaleanchor="x", scaleratio=1, range=[0, 1.0]),
        yaxis=dict(scaleanchor="y", scaleratio=1, range=[0, 1.05]),
        legend=dict(x=0.60, y=0.01, bordercolor="Black", borderwidth=1)
    )
    
    fig.show()

def performance_metrics_plot(
    estimator: EnhancedGridSearchCV,
    variability={
        'name': 'std', 
        'description': 'Standard Deviation',
        'func': np.std
    }
):
    best_metrics = estimator.result_.best.recalculated_scores

    data = {
        metric: {
            'mean': np.mean(data),
            variability['name']: variability['func'](data),
        } for metric, data in best_metrics.items()
    }

    metrics = list(data.keys())

    means = [data[metric]["mean"] for metric in metrics]
    stds = [data[metric][variability['name']] for metric in metrics]

    fig = go.Figure(data=go.Bar(
        x=metrics,
        y=means,
        error_y=dict(type='data', array=stds, visible=True),
        marker_color='lightskyblue'
    ))

    fig.update_layout(
        title=f'Metric Performance with {variability['description']}',
        xaxis_title='Metric',
        yaxis_title='Mean Value',
        template='plotly_white'
    )

    fig.show()
