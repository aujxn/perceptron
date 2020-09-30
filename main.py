import pandas as pd
import json
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.figure_factory as ff
import copy
import itertools
import numpy as np

def scatter(train_set_dfs, test_set_dfs, labels, title):
    scatter_plot = go.Figure()
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for (train, test, label, color) in zip(train_set_dfs, test_set_dfs, labels, colors):
        scatter_plot.add_trace(
                go.Scatter(
                    x=[*range(50)],
                    y=train['accuracy'],
                    line = dict(color=color, dash='dash'),
                    name='training set ' + label
                    )
                )

        scatter_plot.add_trace(
                go.Scatter(
                    x=[*range(50)],
                    y=test['accuracy'],
                    line = dict(color=color),
                    name='testing set ' + label
                    )
                )

    scatter_plot.update_layout(title=title)
    scatter_plot.update_xaxes(title_text='Epoch')
    scatter_plot.update_yaxes(title_text='Accuracy')
    return scatter_plot

def confusion(data, title):
    texts = copy.deepcopy(data)
    data = copy.deepcopy(data)
    for i in range(10):
        data[i][i] = 0

    heatmap = ff.create_annotated_heatmap(
            data, 
            x=[*range(10)],
            y=[*range(10)],
            annotation_text=texts,
            colorscale='Reds'
            )
    heatmap.update_layout(
            title={'text': title, 'yanchor': 'bottom', 'y': 0.1, 'x': 0.5}, 
            yaxis=dict(autorange='reversed'))
    heatmap.update_xaxes(title_text='Predicted')
    heatmap.update_yaxes(title_text='Actual')
    return heatmap

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

with open('./data/output.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

###################################
# Hidden Dimension - Experiment 1 #
###################################

# Filter data to isolate change in hidden layer size for scatter plot
hidden_dimension = df.loc[(df['momentum'] == 0.9) & (df['train_set_size'] == 60000)]

# Seperate test and train results
test_set = hidden_dimension.loc[(hidden_dimension['note'] == 'test set')]
train_set = hidden_dimension.loc[(hidden_dimension['note'] == 'train set')]

# Seperate into 20, 50, and 100 hidden dimension
hidden_dimensions = [20, 50, 100]
labels = ['20 hidden', '50 hidden', '100 hidden']
train_sets = [train_set.loc[df['hidden_dimension'] == var] for var in hidden_dimensions]
test_sets = [test_set.loc[df['hidden_dimension'] == var] for var in hidden_dimensions]

hidden_dimension = scatter(
        train_sets, 
        test_sets,
        labels,
        'Training rate of network with varying hidden layer size'
        )

confusion_matrices = [
        data.loc[(df['epoch'] == 10)] ['confusion_matrix'].values[0] for data in test_sets
        ]
hidden_confusions = [confusion(data, title) for (data, title) in zip(confusion_matrices, labels)]

#################################
# Momentum Value - Experiment 2 #
#################################

# Filter data to isolate change in momentum for scatter plot
momentum = df.loc[(df['hidden_dimension'] == 100) & (df['train_set_size'] == 60000)]

# Seperate test and train results
test_set = momentum.loc[(momentum['note'] == 'test set')]
train_set = momentum.loc[(momentum['note'] == 'train set')]

# Seperate into 0, 0.25, 0.5, and 0.9 momentums
momentums = [0.0, 0.25, 0.5, 0.9]
labels = ['0 momentum', '0.25 momentum', '0.5 momentum', '0.9 momentum']
train_sets = [train_set.loc[df['momentum'] == var] for var in momentums]
test_sets = [test_set.loc[df['momentum'] == var] for var in momentums]

momentums_scatter = scatter(
        train_sets, 
        test_sets,
        labels,
        'Training rate of network with varying momentum'
        )

confusion_matrices = [
        data.loc[(df['epoch'] == 10)] ['confusion_matrix'].values[0] for data in test_sets
        ]
momentum_confusions = [confusion(data, title) for (data, title) in zip(confusion_matrices, labels)]

####################################
# Training Examples - Experiment 3 #
####################################

# Filter data to isolate change in examples for scatter plot
examples = df.loc[(df['hidden_dimension'] == 100) & (df['momentum'] == 0.9)]

# Seperate test and train results
test_set = examples.loc[(examples['note'] == 'test set')]
train_set = examples.loc[(examples['note'] == 'train set')]

# Seperate into 0, 0.25, 0.5, and 0.9 momentums
examples = [15000, 30000, 60000]
labels = ['25% of training set', '50% of training set', '100% of training set']
train_sets = [train_set.loc[df['train_set_size'] == var] for var in examples]
test_sets = [test_set.loc[df['train_set_size'] == var] for var in examples]

examples_scatter = scatter(
        train_sets, 
        test_sets,
        labels,
        'Training rate of network with varying training data set size'
        )

confusion_matrices = [
        data.loc[(df['epoch'] == 10)] ['confusion_matrix'].values[0] for data in test_sets
        ]
examples_confusions = [confusion(data, title) for (data, title) in zip(confusion_matrices, labels)]

# Dash app
app.layout = html.Div(children=[
    html.H1(children='Multilayer Perceptron for MNIST'),

    html.Div([
        dcc.Graph(
            id='hidden_dimension_scatter',
            figure=hidden_dimension
            ),
        ],
        style={'width': '100%', 'display': 'inline-block'},
        ),

    html.Div([
        html.Div([
            dcc.Graph(
                id='hidden_dimension_confusion_20',
                figure=hidden_confusions[0]
                )],
            className="four columns"),
        html.Div([
            dcc.Graph(
                id='hidden_dimension_confusion_50',
                figure=hidden_confusions[1]
                )],
            className="four columns"),
        html.Div([
            dcc.Graph(
                id='hidden_dimension_confusion_100',
                figure=hidden_confusions[2]
                )],
            className="four columns"),
        ],
        style={'width': '100%', 'display': 'inline-block'},
        ),

    html.Div([
        dcc.Graph(
            id='momentums_scatter',
            figure=momentums_scatter
            ),
        ],
        style={'width': '100%', 'display': 'inline-block'},
        ),

    html.Div([
        html.Div([
            dcc.Graph(
                id='momentums_confusion_0',
                figure=momentum_confusions[0]
                )],
            className="six columns"),

        html.Div([
            dcc.Graph(
                id='momentums_confusion_25',
                figure=momentum_confusions[1]
                ),],
            className="six columns")
        ],
        style={'width': '100%', 'display': 'inline-block'}
        ),

    html.Div([
        html.Div([
            dcc.Graph(
                id='momentums_confusion_50',
                figure=momentum_confusions[2]
                )],
            className="six columns"),

            html.Div([
                dcc.Graph(
                    id='momentums_confusion_90',
                    figure=momentum_confusions[3]
                    )],
                className="six columns"),
            ],
            style={'width': '100%', 'display': 'inline-block'},
            ),

    html.Div([
        dcc.Graph(
            id='examples_scatter',
            figure=examples_scatter
            ),
        ],
        style={'width': '100%', 'display': 'inline-block'},
        ),

    html.Div([
        html.Div([
            dcc.Graph(
                id='examples_confusions_25',
                figure=examples_confusions[0]
                )],
            className="four columns"),
        html.Div([
            dcc.Graph(
                id='examples_confusion_50',
                figure=examples_confusions[1]
                )],
            className="four columns"),
        html.Div([
            dcc.Graph(
                id='examples_confusion_100',
                figure=examples_confusions[2]
                )],
            className="four columns"),
        ],
        style={'width': '100%', 'display': 'inline-block'},
        ),
    ],
    style={'width': '80%', 'display': 'inline-block'},
)

if __name__ == '__main__':
    app.run_server(debug=True)
