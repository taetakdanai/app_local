import base64
import io
import joblib
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc
from model_inference_example import BatteryRULPredictor  # Import the class

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize the predictor (using 'zenodo' model)
predictor = BatteryRULPredictor(model_type='zenodo')

# Define the layout of the app
app.layout = dbc.Container(
    [
        html.H2("Battery Remaining Useful Life (RUL) Predictor", className="text-center mt-4 mb-3"),
        html.P("Upload a CSV in the same format as your raw experiment data or engineered cycle data.", className="text-center text-muted mb-4"),
        
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
        html.Div(
            id="status-msg",
            children="Upload a CSV file containing battery cycle data to begin.",
            className="text-center text-muted mb-4",
        ),
        
        # Protocol Dropdown (affects only RUL prediction for selected protocol)
        html.Div(
            [
                html.Label("Select Protocol ID for RUL Prediction:"),
                dcc.Dropdown(
                    id="protocol-dropdown",
                    options=[{"label": f"Protocol {i}", "value": i} for i in range(1, 17)],
                    value=1,  # Default protocol
                    style={"width": "50%", "margin": "auto"},
                ),
            ],
            className="mb-4 text-center",
        ),
        
        # RUL Prediction output card
        dbc.Card(
            dbc.CardBody(
                [
                    html.H3("RUL Prediction for Selected Protocol:", className="card-title text-center"),
                    html.Div(
                        id="prediction-output",
                        children="--",
                        className="text-center text-primary",
                        style={"fontSize": "32px", "fontWeight": "bold"},
                    ),
                ]
            ),
            className="mb-4 shadow-sm",
        ),
        
        # Table for predicted RUL values for all protocols
        html.Div(
            id="protocol-rul-table",
            children=[
                html.H4("Predicted RUL for All Protocols:", className="text-center"),
                DataTable(id='rul-table', style_table={'height': '300px', 'overflowY': 'auto'})
            ],
            className="mb-4",
        ),
    ],
    fluid=True,
)

# Define the callback to process data and provide predictions
@app.callback(
    Output("prediction-output", "children"),
    Output("status-msg", "children"),
    Output("rul-table", "data"),
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
        )

    try:
        # Decode uploaded content
        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)

        # Try UTF-8, then fall back to other encoding
        try:
            s = decoded.decode("utf-8")
        except UnicodeDecodeError:
            s = decoded.decode("ISO-8859-1")

        df_raw = pd.read_csv(io.StringIO(s))

        if df_raw is None or df_raw.empty:
            return "--", "Uploaded file is empty or unreadable.", []

        # Get the last row of the dataset
        last_row = df_raw.iloc[[-1]]  # Select only the last row
        
        # List to store predicted RUL for each protocol
        protocol_rul_list = []

        # Simulate RUL predictions for all protocols (1-16)
        for protocol in range(1, 17):
            # Create a copy of the last row with the current protocol_id
            df_protocol = last_row.copy()
            df_protocol["protocol_id"] = protocol  # Change protocol_id for simulation
            
            # Make predictions using the loaded model (for each protocol)
            predictions = predictor.predict(df_protocol)  # Use the preprocessor from the predictor
            
            # Calculate the mean RUL for this protocol (since we have one row, it is just the prediction)
            mean_rul = predictions[0]
            protocol_rul_list.append({"protocol_id": protocol, "Predicted RUL (Cycles)": mean_rul})

        # Convert the list of protocol RUL results to a DataFrame for the table
        protocol_rul_df = pd.DataFrame(protocol_rul_list)

        # For the selected protocol, we predict based on the last row with the selected protocol_id.
        # This is used for the output card, not the table.
        df_raw["protocol_id"] = selected_protocol  # Modify for selected protocol
        predictions = predictor.predict(df_raw.iloc[[-1]])  # Predict only for the last row

        # Display predicted RUL for the selected protocol
        prediction = predictions[0]  # Assuming we take the first prediction for demo

        status = (
            f"Successfully analyzed file: {filename}. "
            f"Predicted RUL for the selected Protocol {selected_protocol} is {prediction:.2f} cycles."
        )

        # Return the table data, prediction output, and status message
        return f"{prediction:.2f} cycles", status, protocol_rul_df.to_dict('records')

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return (
            "--",
            f"Error processing file: {filename}. Please check file format and required columns. Details: {e}",
            [],
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
