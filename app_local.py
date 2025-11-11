import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc
from model_inference_example import BatteryRULPredictor  # Import the class

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize the predictor (using 'zenodo' model)
predictor = BatteryRULPredictor(model_type='zenodo')

# Define layout
app.layout = dbc.Container(
    [
        html.H2("Battery Remaining Useful Life (RUL) Predictor",
                className="text-center mt-4 mb-3"),
        html.P(
            "Upload a CSV file. The app will use ONLY the last row and simulate RUL predictions for protocol IDs 1–16.",
            className="text-center text-muted mb-4"
        ),

        # Upload button
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select CSV File", className="text-primary")]),
            style={
                "width": "100%",
                "height": "70px",
                "lineHeight": "70px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "8px",
                "textAlign": "center",
                "margin": "15px 0",
                "backgroundColor": "#f9f9f9",
                "cursor": "pointer",
            },
            multiple=False,
        ),

        # Status message
        html.Div(id="status-msg", className="text-center text-muted mb-4"),

        # Dropdown for selected protocol
        html.Div(
            [
                html.Label("Select Protocol ID for RUL Prediction:"),
                dcc.Dropdown(
                    id="protocol-dropdown",
                    options=[{"label": f"Protocol {i}", "value": i} for i in range(1, 17)],
                    value=1,
                    style={"width": "50%", "margin": "auto"},
                ),
            ],
            className="mb-4 text-center",
        ),

        # Prediction card
        dbc.Card(
            dbc.CardBody(
                [
                    html.H3("RUL Prediction for Selected Protocol:",
                            className="card-title text-center"),
                    html.Div(id="prediction-output",
                             children="--",
                             className="text-center text-primary",
                             style={"fontSize": "32px", "fontWeight": "bold"}),
                ]
            ),
            className="mb-4 shadow-sm",
        ),

        # Table of predictions
        html.Div(
            [
                html.H4("Predicted RUL for All Protocols (Simulated from Last Row):",
                        className="text-center"),
                DataTable(
                    id='rul-table',
                    columns=[
                        {"name": "Protocol ID", "id": "protocol_id"},
                        {"name": "Predicted RUL (Cycles)", "id": "Predicted RUL (Cycles)"},
                    ],
                    data=[],
                    style_table={'height': '300px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'center'},
                ),
            ],
            className="mb-4",
        ),

        # Bar chart visualization
        dcc.Graph(id="protocol-rul-bar", style={"height": "450px"}),
    ],
    fluid=True,
)

# Callback
@app.callback(
    Output("prediction-output", "children"),
    Output("status-msg", "children"),
    Output("rul-table", "data"),
    Output("protocol-rul-bar", "figure"),
    Input("upload-data", "contents"),
    Input("protocol-dropdown", "value"),
    State("upload-data", "filename"),
)
def update_output(contents, selected_protocol, filename):
    if contents is None:
        return (
            "--",
            "Upload a CSV file containing battery cycle data to begin.",
            [],
            {},
        )

    try:
        # Decode uploaded CSV
        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        try:
            s = decoded.decode("utf-8")
        except UnicodeDecodeError:
            s = decoded.decode("ISO-8859-1")

        df_raw = pd.read_csv(io.StringIO(s))
        if df_raw.empty:
            return "--", "Uploaded file is empty or unreadable.", [], {}

        # Use only the last row
        last_row = df_raw.iloc[[-1]]

        protocol_rul_list = []

        # Simulate RUL for all protocols
        for protocol in range(1, 17):
            row_copy = last_row.copy()
            row_copy["protocol_id"] = protocol
            preds = predictor.predict(row_copy)
            rul_value = float(preds[0])
            protocol_rul_list.append({
                "protocol_id": protocol,
                "Protocol Label": f"Protocol {protocol}",
                "Predicted RUL (Cycles)": rul_value
            })

        protocol_rul_df = pd.DataFrame(protocol_rul_list)

        # Find selected protocol RUL
        selected_rul = protocol_rul_df.loc[
            protocol_rul_df["protocol_id"] == selected_protocol,
            "Predicted RUL (Cycles)"
        ].values[0]
        prediction_text = f"{selected_rul:.2f} cycles"

        status = (
            f"Successfully analyzed file: {filename}. "
            f"Simulated using the last row for protocols 1–16."
        )

        # Bar chart with distinct colors and labels
        bar_fig = px.bar(
            protocol_rul_df,
            x="Protocol Label",
            y="Predicted RUL (Cycles)",
            color="Protocol Label",  # unique color for each bar
            text="Predicted RUL (Cycles)",
            title="Predicted RUL by Protocol ID (Using Last Row Simulation)",
        )

        bar_fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            hovertemplate="Protocol %{x}<br>RUL: %{y:.2f} cycles<extra></extra>"
        )
        bar_fig.update_layout(
            xaxis_title="Protocol",
            yaxis_title="Predicted RUL (Cycles)",
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )

        return prediction_text, status, protocol_rul_df.to_dict("records"), bar_fig

    except Exception as e:
        print(f"Error: {e}")
        return (
            "--",
            f"Error processing file: {filename}. Details: {e}",
            [],
            {},
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
