import dash
from dash.dependencies import Input, Output
from dash import html
import dash_daq as daq

app = dash.Dash(__name__)

app.layout = html.Div([
    daq.ColorPicker(
        id='my-color-picker-1',
        label='Color Picker',
        value=dict(hex='#119DFF')
    ),
    html.Div(id='color-picker-output-1')
])

@app.callback(
    Output('color-picker-output-1', 'children'),
    Input('my-color-picker-1', 'value')
)
def update_output(value):
    return 'The selected color is {}.'.format(value)

if __name__ == '__main__':
    app.run_server(debug=False)