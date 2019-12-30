#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import display, IFrame, HTML
import os

def show_app(app, port=9999, width=900, height=700):
    host = 'localhost'
    url = f'http://{host}:{port}'

    display(HTML(f"<a href='{url}' target='_blank'>Open in new tab</a>"))
    display(IFrame(url, width=width, height=height))
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True
    return app.run_server(debug=False, host=host, port=port)

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pickle

from scipy.spatial.distance import euclidean, cityblock, cosine
from dash.dependencies import Input, Output, State

app = dash.Dash(assets_folder='web/data', assets_url_path='web/data')

with open('data/models1daw.pickle', 'rb') as fp:
    models1daw = pickle.load(fp)
    
# Questions mapping
# Where do you live?
y5_mapping = {'Rural': 0, 'Urban': 1}
# What is your marital status?
a1_mapping = {'Single': 0, 'Married/Live-in': 1}
# Are you religious?
a15_mapping = {'Yes': 1, 'No': 0}
# Do you get along with your siblings well? (Higher is More Prone)
b2_mapping = {'Yes, all of them': 0, 'Yes, at least one but not all': 1, 'No': 2}
# Do your parents get along well? (Higher is More Prone)
b13_mapping = {'Most of the time': 1, 
               'Sometimes': 2, 
               'All the time': 0, 
               'Never': 3}
# How do you get along with your father? (Higher is More Prone)
b14_mapping = {'Most of the time': 1, 
                 'All the time': 0, 
                 'Sometimes': 2, 
                 'Never': 3}
# How do you get along with your mother? (Higher is More Prone)
b15_mapping = {'Most of the time': 1, 
                 'All the time': 0, 
                 'Sometimes': 2, 
                 'Never': 3}
# You share your problems more to friends rather than family (Higher is More Prone)
b28b_mapping = {'Sometimes': 2, 'Once in a while': 1, 'Frequently': 3, 
                 'Never': 0, 'Always': 4}
# Were you disciplined in the family? (Higher is More Prone)
b28d_mapping = {'Sometimes': 2, 'Once in a while': 3, 'Frequently': 1, 
                 'Never': 4, 'Always': 0}
# Were you verbally abused by your family?
b31a5_mapping = {'Yes': 1, 'No': 0}
# Were you physically abused by your family?
b31a6_mapping = {'Yes': 1, 'No': 0}      
# How would you rate your happpiness? (Higher is better)
c5_mapping = {'Very happy': 10, 'Not happy at all': 1, '2': 2, '3': 3, '4': 4,
             '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
# How is your schooling?
d1_mapping = {'In school': 0, 
              'Currently not in school': 1, 
              'Never been in school': 2}
# Are you involved in a fraternity or sorority?
d214_mapping = {'Yes': 1, 'No': 0}
# Do you read pornography?
e18_mapping = {'Yes': 1, 'No': 0}
# Do you watch pornography?
e21_mapping = {'Yes': 1, 'No': 0}
# Do you ever feel depressed?
g1802_mapping = {'Sometimes': 1, 'Rarely': 0, 'Often': 2}
# Do you smoke?
g19_mapping = {'Yes': 1, 'No': 0}
# Do you drink?
g30_mapping = {'Yes': 1, 'No': 0}
# Do you have a family member who uses drugs?
g73_mapping = {'Yes': 1, 'No': 0}
# Were you physically abused in general?
g77_mapping = {'Yes': 1, 'No': 0}
# Do you have suicidal thoughts?
g83_mapping = {'No': 0, 'Once': 1, 'More than once': 2}
# Have you ever been suspended in school?
d23_mapping = {'Yes': 1, 'No': 0}
    
app.callback_map = {}
app.layout = html.Div([
    # Title
    html.Div([
        html.Div('Prevention is Better than Cure:', 
                 style={'color': '#ffc000', 'font-family': 'Arial', 
                        'text-align': 'center', 
                        'font-weight': 'bold', 'font-size': '48px', 
                        'font-size': '3vw', 'width': '75%',
                        'margin': 'auto'}),
        html.Div('Predicting Illegal Drug Vulnerability among Filipino Youth', 
                 style={'font-family': 'Arial', 'text-align': 'center', 
                        'font-weight': 'bold', 'font-size': '42px', 
                        'font-size': '2.5vw', 'width': '75%',
                        'margin': 'auto'})
    ], style={'background-image': 'url(web/data/title.JPG)',
              'height': '140px',
              'margin': 'auto'}),
    # Sliders
    html.Div([
        html.Br(),
        html.Br(),
        html.Div(['''The problem of illegal drugs is one of the most pressing 
        issues in the Philippines for many lives are affected by it. This 
        website aims to estimate the propensity for illegal drug use of a 
        Filipino young adult using Machine Learning.
        '''], style={'font-size': '22px', 'margin': 'auto', 
                     'text-align': 'justify'}),
        html.Br(),
        html.Br(),
        html.Div(['''The dataset was sourced 
        from the 2013 Young Adult Fertility and Sexuality Study (YAFS) 
        with a total of 19,728 respondents. Among the various machine 
        learning models that we evaluated, we found Gradient Boosting 
        Classifier to yield the highest accuracy of 95.4%.
        '''], style={'font-size': '22px', 'margin': 'auto', 
                     'text-align': 'justify'}),
        html.Br(),
        html.Br(),
        html.Div(['''We envisage that 
        with improved identification of vulnerable youth, necessary 
        interventions can be introduced more quickly and more accurately.
        '''], style={'font-size': '22px', 'margin': 'auto', 
                     'text-align': 'justify'}),
        html.Br(),
        html.Br(),
        html.Div(['''You can now calculate your vulnerability to illegal drugs by
        answering the survey below!
        '''], style={'font-size': '22px', 'margin': 'auto', 
                     'text-align': 'justify'}),
        html.Br(),
        html.Br(),
        html.Hr(),
        html.Div('Demographic', style={'font-weight': 'bold',
                                       'font-size': '24px',
                                       'color': '#ffc000'}),
        html.Div([
            html.Div([
                html.Br(),
                html.Br(),
                html.Label('Are you religious?'),
                dcc.Slider(id='c',
                min=0,
                max=1,
                marks={y:x for x, y in a15_mapping.items()},
                value=0,
                updatemode='mouseup'
            )], style={'width': '75%', 'margin': 'auto'}),
            html.Div([
                html.Br(),
                html.Br(),
                html.Label('Where do you live?'),
                dcc.Slider(id='a',
                min=0,
                max=1,
                marks={y:x for x, y in y5_mapping.items()},
                value=0,
                updatemode='mouseup'
            )], style={'width': '75%', 'margin': 'auto'}),
            html.Div([
                html.Br(),
                html.Br(),
                html.Label('What is your marital status?'),
                dcc.Slider(id='b',
                min=0,
                max=1,
                marks={y:x for x, y in a1_mapping.items()},
                value=0,
                updatemode='mouseup'
            )], style={'width': '75%', 'margin': 'auto'}),
            html.Br(),
            html.Br(),
        ])
    ], style={'width': '75%', 'margin': 'auto'}),
    html.Div([
        html.Hr(),
        html.Div('Family Relationship', style={'font-weight': 'bold',
                                               'font-size': '24px',
                                               'color': '#ffc000'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Were you verbally abused by your family?'),
            dcc.Slider(id='j',
            min=0,
            max=1,
            marks={y:x for x, y in b31a5_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Were you physically abused by your family?'),
            dcc.Slider(id='k',
            min=0,
            max=1,
            marks={y:x for x, y in b31a6_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do you get along with your siblings well?'),
            dcc.Slider(id='d',
            min=0,
            max=2,
            marks={y:x for x, y in b2_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),    
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do your parents get along well?'),
            dcc.Slider(id='e',
            min=0,
            max=3,
            marks={y:x for x, y in b13_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do you get along with your father?'),
            dcc.Slider(id='f',
            min=0,
            max=3,
            marks={y:x for x, y in b14_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do you get along with your mother?'),
            dcc.Slider(id='g',
            min=0,
            max=3,
            marks={y:x for x, y in b15_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Were you disciplined in the family?'),
            dcc.Slider(id='i',
            min=0,
            max=4,
            marks={y:x for x, y in b28d_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('You share your problems more to friends rather than family'),
            dcc.Slider(id='h',
            min=0,
            max=4,
            marks={y:x for x, y in b28b_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Br(),
        html.Br(),
        html.Hr()   
    ], style={'width': '75%', 'margin': 'auto'}),
    html.Div([
        html.Div('Activity', style={'font-weight': 'bold',
                                    'font-size': '24px',
                                    'color': '#ffc000'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do you smoke?'),
            dcc.Slider(id='r',
            min=0,
            max=1,
            marks={y:x for x, y in g19_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do you drink?'),
            dcc.Slider(id='s',
            min=0,
            max=1,
            marks={y:x for x, y in g30_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do you read pornography?'),
            dcc.Slider(id='o',
            min=0,
            max=1,
            marks={y:x for x, y in e18_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do you watch pornography?'),
            dcc.Slider(id='p',
            min=0,
            max=1,
            marks={y:x for x, y in e21_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Are you involved in a fraternity or sorority?'),
            dcc.Slider(id='n',
            min=0,
            max=1,
            marks={y:x for x, y in d214_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('How is your schooling?'),
            dcc.Slider(id='m',
            min=0,
            max=2,
            marks={y:x for x, y in d1_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do you ever feel depressed?'),
            dcc.Slider(id='q',
            min=0,
            max=2,
            marks={y:x for x, y in g1802_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('How would you rate your happpiness?'),
            dcc.Slider(id='l',
            min=1,
            max=10,
            marks={y:x for x, y in c5_mapping.items()},
            value=1,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
    ], style={'width': '75%', 'margin': 'auto'}),
    html.Div([
        html.Br(),
        html.Br(),
        html.Hr(),
        html.Div('History', style={'font-weight': 'bold',
                                   'font-size': '24px',
                                   'color': '#ffc000'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do you have a family member who uses drugs?'),
            dcc.Slider(id='t',
            min=0,
            max=1,
            marks={y:x for x, y in g73_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Were you physically abused in general?'),
            dcc.Slider(id='u',
            min=0,
            max=1,
            marks={y:x for x, y in g77_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Have you ever been suspended in school?'),
            dcc.Slider(id='w',
            min=0,
            max=1,
            marks={y:x for x, y in d23_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
        html.Div([
            html.Br(),
            html.Br(),
            html.Label('Do you have suicidal thoughts?'),
            dcc.Slider(id='v',
            min=0,
            max=2,
            marks={y:x for x, y in g83_mapping.items()},
            value=0,
            updatemode='mouseup'
        )], style={'width': '75%', 'margin': 'auto'}),
    ], style={'width': '75%', 'margin': 'auto'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    # Check button
    html.Div(
        [html.Button('Calculate!', id='button')], 
        style={'text-align': 'center'}),
    html.Br(),
    # Prediction
    html.Div(id='pred', style={'text-align': 'center', 'font-size': '36px',
                               'font-weight': 'bold', 'color': '#ffc000'}),
    html.Br(),
    html.Br(),
    html.Div(html.Hr(), style={'margin': 'auto', 'width': '75%'}),
    html.Br(),
    html.Br(),
    html.Div('Our Methodology', style={'text-align': 'center',
                                       'font-size': '36px',
                                       'font-weight': 'bold',
                                       'color': 'white'}),
    html.Br(),
    html.Div([
        html.Img(src=app.get_asset_url('methodology.JPG'),
                         style={'max-width': '65%', 'max-height': '100%'})
    ], style={'text-align': 'center'})
    ], style={'font-family': 'Arial', 'color': 'white', 
              'background-color': 'black', 'padding': '0px'})

@app.callback(Output('pred', 'children'),
              [Input('button', 'n_clicks')],
              [State('a', 'value'), State('b', 'value'), State('c', 'value'), 
               State('d', 'value'), State('e', 'value'), State('f', 'value'),
               State('g', 'value'), State('h', 'value'), State('i', 'value'),
               State('j', 'value'), State('k', 'value'), State('l', 'value'),
               State('m', 'value'), State('n', 'value'), State('o', 'value'),
               State('p', 'value'), State('q', 'value'), State('r', 'value'),
               State('s', 'value'), State('t', 'value'), State('u', 'value'),
               State('v', 'value'), State('w', 'value')])
def display_prediction(n_clicks, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o,
                      p, q, r, s, t, u, v, w):
    if n_clicks:
        return 'You are {:.2f}% vulnerable to drugs!'.format(models1daw['Gradient Boosting Classifier'].best_estimator_.                    predict_proba([[a, b, c, d, e, f, g, h, i, j, k, l, m, n, 
                                    o, p, q, r, s, t, u, v, w]])[0][1]*100)

if __name__ == '__main__':
    app.run_server(debug=False)
    
show_app(app)


# In[ ]:




