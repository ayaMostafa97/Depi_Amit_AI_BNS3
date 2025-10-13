
"""
🏠 House Price Analytics Dashboard
Updated version with Linear Regression model using 3 features.
Attractive design implemented, and Callback logic separated for increased stability.
The issue with updating the price-vs-feature-graph has been resolved.
Run: python app.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# ====== Data Setup and Model Initialization ======
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "C:/Users/Khaled/Depi_Amit_A1_BNS3/Depi_Amit_A1_BNS3/First_ML_Project/housing_dashboard\housing.csv"

# Required features for this version (RM, LSTAT, PTRATIO)
TARGET_COL = "MEDV" 
FEATURE_ORDER = ["RM", "LSTAT", "PTRATIO"]
ALL_FEATURES = FEATURE_ORDER + [TARGET_COL]

def create_synthetic_data():
    """Creates synthetic data simulating Boston house features (RM, LSTAT, PTRATIO)"""
    # NOTE: Removed np.random.seed(42) to allow metrics to change slightly on retrain
    # ملاحظة: تم إزالة np.random.seed(42) للسماح بتغيير المقاييس قليلاً عند إعادة التدريب
    n = 506
    df = pd.DataFrame({
        "RM": np.random.normal(6.3, 0.7, n).clip(3, 9), 
        "LSTAT": np.random.normal(12.7, 5.5, n).clip(1, 40), 
        "PTRATIO": np.random.normal(18.5, 2.0, n).clip(12, 22), 
    })
    # Simple equation to generate a price based on the three features
    # --- START OF MODIFICATION ---
    # لقد قمت بتغيير المعاملات قليلاً هنا (مثلاً: RM * 12 بدلاً من * 10)
    df[TARGET_COL] = (
        df["RM"] * 12 - df["LSTAT"] * 0.4 + 
        (25 - df["PTRATIO"]) * 3 + 
        np.random.normal(0, 4, n) + 25
    ).round(2).clip(10, 50) # Prices simulate original MEDV values
    # --- END OF MODIFICATION ---

    return df.rename(columns={col: col.upper() for col in df.columns})

def load_dataset():
    """Loads data from file or creates synthetic data"""
    # Ensure synthetic data is used as a reliable fallback
    df = create_synthetic_data() 
    if DATA_PATH.exists():
        try:
            df_file = pd.read_csv(DATA_PATH)
            df_file.columns = [c.strip().upper() for c in df_file.columns]
            if all(col in df_file.columns for col in ALL_FEATURES):
                # If file data is valid, use it
                return df_file
        except Exception:
            # If file loading fails or is corrupted, fall back to synthetic data
            print("⚠️ Failed to load housing.csv. Synthetic data will be used.")
            pass
    
    # If file doesn't exist or is invalid, use the synthetic data created initially
    return df

def train_model(df):
    """Trains the Linear Regression model and extracts metrics"""
    # Ensure required columns exist and fill NaN if any
    df_clean = df[ALL_FEATURES].fillna(df[ALL_FEATURES].mean())
    
    y = df_clean[TARGET_COL]
    X = df_clean[FEATURE_ORDER]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds)
    }
    
    # Add model results to the global data
    df['Predicted_MEDV'] = pipe.predict(df[FEATURE_ORDER].fillna(df[FEATURE_ORDER].mean()))
    df['Actual_MEDV'] = df[TARGET_COL]
    
    # Create correlation matrix for analysis purposes
    cols_for_corr = FEATURE_ORDER + ["Actual_MEDV", "Predicted_MEDV"]
    corr_matrix = df[cols_for_corr].corr().round(2)

    return pipe, metrics, corr_matrix

# ====== Load Data and Initial Model ======
df_global = load_dataset()
MODEL_PATH = BASE_DIR / "house_model_lr.joblib" 

MODEL = None
INITIAL_METRICS = {}
INITIAL_CORR_MATRIX = pd.DataFrame()

try:
    # 1. Attempt to load the saved model
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
        # We need to retrain the model at least once to ensure 'Predicted_MEDV' and 'Actual_MEDV' columns 
        # exist in df_global and to get the initial correlation matrix.
        _, INITIAL_METRICS, INITIAL_CORR_MATRIX = train_model(df_global) 
    else:
        # 2. Train and save the model if it does not exist
        MODEL, INITIAL_METRICS, INITIAL_CORR_MATRIX = train_model(df_global)
        joblib.dump(MODEL, MODEL_PATH)
except Exception as e:
    # 3. If loading or training fails, create a new model and save it
    print(f"❌ Error loading/training model: {e}. A new model will be created.")
    MODEL, INITIAL_METRICS, INITIAL_CORR_MATRIX = train_model(df_global)
    joblib.dump(MODEL, MODEL_PATH)
    
INITIAL_METRICS_MSG = f"ℹ️ Ready — Linear Regression Model loaded. MAE={INITIAL_METRICS.get('mae', 0):.2f}, RMSE={INITIAL_METRICS.get('rmse', 0):.2f}, R²={INITIAL_METRICS.get('r2', 0):.3f}"


# ====== Build Dash Interface ======
app = dash.Dash(__name__)
app.title = "House Price Analytics Dashboard"

# Helper function to generate the correlation table using dash_table.DataTable
def generate_correlation_table(df_corr):
    """Converts the correlation matrix into a Dash DataTable"""
    df_corr_reset = df_corr.reset_index().rename(columns={'index': 'Feature'})
    
    # Format numbers within the table only for display
    display_data = df_corr_reset.copy()
    for col in display_data.columns[1:]:
        # Ensure columns are numeric before formatting
        display_data[col] = display_data[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

    return dash_table.DataTable(
        id='correlation-table',
        columns=[{"name": i, "id": i} for i in display_data.columns],
        data=display_data.to_dict('records'),
        style_header={
            'backgroundColor': '#e5e7eb',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell={
            'textAlign': 'center',
            'fontFamily': 'Arial, sans-serif',
            'padding': '10px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': 'auto', 'padding': '20px', 'backgroundColor': '#f9f9f9'}, children=[
    
    html.H2("🏠 House Price Analytics and Prediction Dashboard (Linear Regression)", style={"textAlign": "center", "marginBottom": "25px", 'color': '#1f2937'}),

    # Control and Prediction Section
    html.Div([
        html.Button("Retrain Model", id="btn-retrain", n_clicks=0, 
                    style={"marginRight": "10px", 'padding': '10px 20px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'cursor': 'pointer', 'transition': 'all 0.3s'}),
        html.Button("Predict Price", id="btn-predict", n_clicks=0, 
                    style={"backgroundColor": "#10b981", "color": "white", 'padding': '10px 20px', 'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer', 'transition': 'all 0.3s'})
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Metrics and Status Message
    html.Div(id="kpi-metrics", children=[INITIAL_METRICS_MSG], style={"textAlign": "center", "marginTop": "10px", 'padding': '15px', 'backgroundColor': '#eef2ff', 'borderRadius': '8px', 'marginBottom': '20px', 'border': '1px solid #c7d2fe'}),

    # Feature Input Fields (3 columns)
    html.Div([
        html.Div([
            html.Label("RM (Average Rooms)", style={'fontWeight': 'bold', 'color': '#374151'}),
            dcc.Input(id="in-rm", type="number", value=6.5, placeholder="Average number of rooms", min=3, max=9, step='0.1', style={'width': '100%', 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '6px', 'marginTop': '5px'}),
        ], style={'padding': '10px'}),
        
        html.Div([
            html.Label("LSTAT (Lower Status Ratio)", style={'fontWeight': 'bold', 'color': '#374151'}),
            dcc.Input(id="in-lstat", type="number", value=10, placeholder="LSTAT Ratio", min=1, max=40, step='0.1', style={'width': '100%', 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '6px', 'marginTop': '5px'}),
        ], style={'padding': '10px'}),
        
        html.Div([
            html.Label("PTRATIO (Pupil/Teacher Ratio)", style={'fontWeight': 'bold', 'color': '#374151'}),
            dcc.Input(id="in-ptratio", type="number", value=15, placeholder="PTRATIO Ratio", min=12, max=22, step='0.1', style={'width': '100%', 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '6px', 'marginTop': '5px'}),
        ], style={'padding': '10px'}),
        
    ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "20px", "margin": "10px 0", 'border': '1px solid #eee', 'borderRadius': '8px', 'backgroundColor': '#fff', 'padding': '10px'}, id="input-container"),

    # Prediction Result
    html.Div(id="pred-result", children=['Enter details and click "Predict Price"'], 
             style={"textAlign": "center", "fontWeight": "bold", "fontSize": "28px", 'color': '#ef4444', 'margin': '30px 0', 'padding': '15px', 'borderBottom': '3px solid #fca5a5'}),

    html.H3("📊 Model Data Analytics", style={"textAlign": "center", "marginTop": "30px", "color": "#374151", 'borderBottom': '1px solid #ddd', 'paddingBottom': '10px'}),

    # Top Graphs Section (Price Distribution - Price vs Feature)
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
        
        # 1. Price Distribution
        html.Div(style={'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': 'white', 'padding': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'}, children=[
            html.H4("Actual Price Distribution", style={'textAlign': 'center', 'color': '#374151'}),
            dcc.Loading(type="circle", children=[
                dcc.Graph(id="price-distribution-graph")
            ])
        ]),

        # 2. Price vs Feature
        html.Div(style={'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': 'white', 'padding': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'}, children=[
            html.H4("Price vs Selected Feature", style={'textAlign': 'center', 'color': '#374151'}),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': f'{f}', 'value': f} for f in FEATURE_ORDER],
                value='LSTAT', # Default value
                clearable=False,
                style={'marginBottom': '10px'}
            ),
            dcc.Loading(type="circle", children=[
                dcc.Graph(id="price-vs-feature-graph") 
            ]),
             # Invisible component to store prediction point data
            dcc.Store(id='prediction-data-store', data={'RM': None, 'LSTAT': None, 'PTRATIO': None, 'predicted_price': None}),
        ])
    ]),
    
    # Bottom Graphs Section (Correlation Heatmap - Correlation Table)
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
        
        # 3. Correlation Heatmap
        html.Div(style={'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': 'white', 'padding': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'}, children=[
            html.H4("Correlation Matrix (Heatmap)", style={'textAlign': 'center', 'color': '#374151'}),
            dcc.Loading(type="circle", children=[
                dcc.Graph(id="correlation-heatmap")
            ])
        ]),

        # 4. Correlation Table
        html.Div(style={'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': 'white', 'padding': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'}, children=[
            html.H4("Correlation Values Table", style={'textAlign': 'center', 'marginBottom': '10px', 'color': '#374151'}),
            dcc.Loading(type="circle", children=[
                html.Div(id="correlation-table-output")
            ])
        ])
    ])
])

# ====== Updated Callbacks for Increased Stability ======

# 1. Callback for Retraining Model and updating Analytics (Metrics, Heatmap, Table, Distribution)
@app.callback(
    Output("kpi-metrics", "children"),
    Output("correlation-heatmap", "figure"),
    Output("correlation-table-output", "children"),
    Output("price-distribution-graph", "figure"),
    Input("btn-retrain", "n_clicks")
)
def retrain_model_and_update_analytics(n_clicks):
    """
    Handles model retraining and updates metrics, correlation plots, and price distribution.
    """
    global MODEL, df_global, INITIAL_METRICS_MSG, INITIAL_CORR_MATRIX
    
    # Check if the Retrain button was clicked
    if dash.callback_context.triggered and dash.callback_context.triggered[0]['prop_id'] == 'btn-retrain.n_clicks':
        MODEL, metrics, corr_matrix = train_model(df_global)
        joblib.dump(MODEL, MODEL_PATH)
        msg = f"✅ Linear Regression Model retrained! MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}"
    else:
         # This runs on initial load, using saved values
         metrics = INITIAL_METRICS
         corr_matrix = INITIAL_CORR_MATRIX
         msg = INITIAL_METRICS_MSG


    # 1. Create Heatmap and Table
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto", title="Correlation Matrix",
        color_continuous_scale=px.colors.diverging.RdBu, zmin=-1, zmax=1
    )
    table = generate_correlation_table(corr_matrix)
    
    # 2. Draw Price Distribution
    if df_global.empty:
        fig_dist = go.Figure()
        fig_dist.update_layout(title="Actual Price Distribution (No Data Available)", height=400)
    else:
        fig_dist = px.histogram(
            df_global, x=TARGET_COL, nbins=20, 
            title="Actual Price Distribution", labels={TARGET_COL: "Price Value (MEDV)"},
            color_discrete_sequence=['#4c78a8']
        )
        fig_dist.update_layout(showlegend=False)

    return msg, fig_corr, table, fig_dist

# 2. Callback for drawing the base historical data plot
@app.callback(
    Output("price-vs-feature-graph", "figure"),
    Input("feature-dropdown", "value")
)
def draw_base_scatter_plot(selected_feature):
    """Draws a scatter plot of Price vs. the selected feature (historical data only)"""
    if not selected_feature or df_global.empty:
        # If no feature is selected or data is empty, return an empty plot
        fig = go.Figure()
        fig.update_layout(title="Price vs Selected Feature (No Data Available)", height=400)
        return fig
        
    fig = px.scatter(
        df_global, 
        x=selected_feature, 
        y=TARGET_COL, 
        title=f"Actual Price vs {selected_feature}",
        labels={selected_feature: selected_feature, TARGET_COL: "Price Value (MEDV)"},
        trendline="ols", 
        color_discrete_sequence=['#f59e0b'],
        height=400 
    )
    
    # Set Y-axis range suitable for MEDV (typically up to 50)
    max_y = df_global[TARGET_COL].max()
    fig.update_yaxes(range=[0, max_y * 1.1 if max_y > 0 else 55])
    
    return fig

# 3. Callback to calculate prediction and store data
@app.callback(
    Output("pred-result", "children"),
    Output("prediction-data-store", "data"),
    Input("btn-predict", "n_clicks"),
    State("in-rm", "value"),
    State("in-lstat", "value"),
    State("in-ptratio", "value"),
    prevent_initial_call=True
)
def calculate_prediction_and_store_data(n, rm, lstat, ptratio):
    """
    Calculates the prediction and stores it in the dcc.Store component.
    """
    if not MODEL:
        error_text = html.Span("❌ Prediction Error. Model not initialized.", style={'color': 'red'})
        return error_text, {'RM': None, 'LSTAT': None, 'PTRATIO': None, 'predicted_price': None}
    
    # Check that all input values are valid (not None)
    if any(val is None for val in [rm, lstat, ptratio]):
        error_text = html.Span("❌ Error: All feature values must be entered.", style={'color': 'red'})
        return error_text, {'RM': None, 'LSTAT': None, 'PTRATIO': None, 'predicted_price': None}

    try:
        input_values = {
            "RM": float(rm), 
            "LSTAT": float(lstat), 
            "PTRATIO": float(ptratio)
        }
        
        # 1. Calculate prediction
        data_point = pd.DataFrame([input_values])[FEATURE_ORDER]
        
        predicted_price = MODEL.predict(data_point)[0]
        
        # 2. Prepare text result
        result_text = html.Span([
            "💰 Predicted MEDV: ",
            html.Span(f"{predicted_price:,.2f}", style={'color': '#10b981', 'fontWeight': 'extrabold'})
        ])
        
        # 3. Store data needed to add the point to the graph
        store_data = {
            'RM': input_values['RM'],
            'LSTAT': input_values['LSTAT'],
            'PTRATIO': input_values['PTRATIO'],
            'predicted_price': predicted_price
        }

        return result_text, store_data
        
    except Exception as e:
        # Print the error to the console for easier debugging
        print(f"Prediction Error: {e}")
        error_text = html.Span("❌ Prediction Error. Check the input values.", style={'color': 'red'})
        return error_text, {'RM': None, 'LSTAT': None, 'PTRATIO': None, 'predicted_price': None}

# 4. Callback to add the prediction point to the graph
@app.callback(
    Output("price-vs-feature-graph", "figure", allow_duplicate=True),
    Input("prediction-data-store", "data"),
    Input("feature-dropdown", "value"), 
    State("price-vs-feature-graph", "figure"),
    prevent_initial_call=True
)
def add_prediction_point_to_graph(data, selected_feature, existing_figure):
    """
    Adds the new prediction point to the existing or updated graph.
    """
    # Initial check to ensure essential data and selected feature exist
    if not data or data['predicted_price'] is None or selected_feature not in data:
        raise PreventUpdate

    # Extract X and Y values for the new point
    x_value = data[selected_feature]
    y_value = data['predicted_price']

    # Use the existing figure object
    if not existing_figure:
        fig = draw_base_scatter_plot(selected_feature)
    else:
        fig = go.Figure(existing_figure)

    # 1. Remove any previous prediction traces to prevent point accumulation
    new_data = [trace for trace in fig.data if not (trace.name and trace.name.startswith('New Prediction'))]
    fig.data = new_data
    
    # 2. Add the new point as a Trace
    fig.add_trace(go.Scatter(
        x=[x_value],
        y=[y_value],
        mode='markers',
        marker=dict(size=15, color='#ef4444', symbol='star-diamond', line=dict(width=2, color='DarkRed')),
        name=f'New Prediction ({selected_feature})',
        hovertemplate=f"<b>{selected_feature}:</b> {x_value}<br><b>Predicted Price:</b> {y_value:,.2f}<extra></extra>"
    ))
    
    # 3. Update labels if the selected feature has changed
    fig.update_layout(
        title=f"Actual Price vs {selected_feature}",
        xaxis_title=selected_feature
    )

    return fig
# ====== Extra Graphs Section (added safely without breaking existing code) ======

extra_graphs_section = html.Div([
    html.H3("🌐 Additional Visual Insights", 
            style={"textAlign": "center", "marginTop": "40px", "color": "#1f2937"}),

    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[

        # 1️⃣ 3D Scatter Plot
        html.Div(style={'border': '1px solid #ddd', 'borderRadius': '10px', 
                        'backgroundColor': 'white', 'padding': '15px', 
                        'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'}, children=[
            html.H4("3D Relationship Between RM, LSTAT, PTRATIO, and MEDV",
                    style={'textAlign': 'center', 'color': '#374151'}),
            dcc.Loading(type="circle", children=[
                dcc.Graph(
                    id="three-d-graph",
                    figure=px.scatter_3d(
                        df_global,
                        x="RM", y="LSTAT", z="PTRATIO",
                        color="MEDV",
                        color_continuous_scale="Viridis",
                        title="3D Scatter: RM vs LSTAT vs PTRATIO (Colored by MEDV)",
                        height=500
                    )
                )
            ])
        ]),

        # 2️⃣ Box Plot of Prices
        html.Div(style={'border': '1px solid #ddd', 'borderRadius': '10px', 
                        'backgroundColor': 'white', 'padding': '15px', 
                        'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'}, children=[
            html.H4("Price Distribution (Box Plot)", 
                    style={'textAlign': 'center', 'color': '#374151'}),
            dcc.Loading(type="circle", children=[
                dcc.Graph(
                    id="price-boxplot",
                    figure=px.box(
                        df_global,
                        y="MEDV",
                        points="all",
                        title="Box Plot of MEDV (House Prices)",
                        color_discrete_sequence=["#3b82f6"],
                        height=500
                    )
                )
            ])
        ])
    ])
])

# Append the extra graphs safely to the existing layout
app.layout.children.append(extra_graphs_section)

if __name__ == "__main__":
    app.run(debug=True)