# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:44:27 2021

@author: mhoum
"""

import dash
import dash_core_components as dcc
import dash_html_components as html

list_of_data = [{
    'y':[654,643,126,987],
    'x':['Nice', 'Paris', 'Toulon', 'Dijon'],
    'type':'bar'
    
    }]

appb = dash.Dash(__name__)

body = html.Div([
    html.H2("Répartition des déchets en France"), html.H2("Aujourd'hui"),
    dcc.Graph(id='s',
              figure= {'data':list_of_data})
    ])

appb.layout = html.Div([body])

if __name__ == "__main__":
    appb.run_server(debug=True,port=3307)