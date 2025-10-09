
from flask import Flask, render_template, send_file
import pandas as pd
import numpy as np
import json, os

app = Flask(__name__)

# ✅ Detect CSV file automatically
folder = os.path.dirname(__file__)
csv_candidates = [
    "C:/Users/Khaled/Depi_Amit_A1_BNS3/Depi_Amit_A1_BNS3/First_ML_Project/flask_dashboard\housing.csv"
]
DATA_PATH = None
for c in csv_candidates:
    path = os.path.join(folder, c)
    if os.path.exists(path):
        DATA_PATH = path
        break

if DATA_PATH:
    df = pd.read_csv(DATA_PATH, low_memory=False)
else:
    df = pd.DataFrame()

# ✅ Convert numeric columns safely
for c in df.select_dtypes(include=['int', 'float']).columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

def filter_df():
    return df

@app.route("/")
def index():
    d = filter_df()
    total_rows = len(d)

    # ✅ Detect target (price) column automatically
    target = None
    for cand in ['MEDV', 'Price', 'price', 'SalePrice', 'median_house_value']:
        if cand in d.columns:
            target = cand
            break

    numeric_cols = d.select_dtypes(include='number').columns.tolist()
    if not target and numeric_cols:
        target = numeric_cols[0]

    # ✅ Basic stats
    avg_price = round(d[target].mean(), 2) if target in d.columns and total_rows > 0 else 0
    median_price = round(d[target].median(), 2) if target in d.columns and total_rows > 0 else 0

    # ✅ Histogram
    hist_values, hist_bins = [], []
    if target in d.columns:
        hist_series = d[target].dropna()
        hist_counts, bin_edges = np.histogram(hist_series, bins=20)
        hist_values = hist_counts.tolist()
        hist_bins = [str(round(b, 2)) for b in bin_edges]

    # ✅ Scatter plot
    scatter_x, scatter_y, scatter_x_label = [], [], None
    if target and len(numeric_cols) > 1:
        other = [c for c in numeric_cols if c != target][0]
        scatter_x = d[other].fillna(0).astype(float).tolist()
        scatter_y = d[target].fillna(0).astype(float).tolist()
        scatter_x_label = other

    # ✅ Correlation matrix
    corr_matrix = {}
    if not d.empty:
        corr_matrix = d.select_dtypes(include='number').corr().round(2).fillna(0).to_dict()

    # ✅ Map markers (if lat/lon exist)
    markers = []
    lat_cols = [c for c in d.columns if 'lat' in c.lower()]
    lon_cols = [c for c in d.columns if 'lon' in c.lower() or 'lng' in c.lower()]

    if lat_cols and lon_cols and target in d.columns:
        latc, lonc = lat_cols[0], lon_cols[0]
        top = d[[latc, lonc, target]].dropna().head(100)
        markers = top.to_dict(orient='records')

    return render_template(
        "index.html",
        total_rows=total_rows,
        avg_price=avg_price,
        median_price=median_price,
        target=target,
        hist_values=json.dumps(hist_values),
        hist_bins=json.dumps(hist_bins),
        scatter_x=json.dumps(scatter_x),
        scatter_y=json.dumps(scatter_y),
        scatter_x_label=scatter_x_label,
        corr_matrix=json.dumps(corr_matrix),
        markers=json.dumps(markers),
        numeric_cols=json.dumps(numeric_cols)
    )

@app.route("/download")
def download():
    if DATA_PATH and os.path.exists(DATA_PATH):
        return send_file(DATA_PATH, as_attachment=True)
    return "No data file found."

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
