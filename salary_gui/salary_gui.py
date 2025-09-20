import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# ----------------------------
# Helper functions
# ----------------------------
def calc_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# ----------------------------
# Title & Sidebar
# ----------------------------
st.title("📌 Salary Prediction - Model Explorer")

st.sidebar.subheader("🔧 Model Configuration")
chosen_models = st.sidebar.multiselect(
    "Which models do you want to include?",
    ["Linear", "Polynomial", "RandomForest", "NeuralNet", "Lasso", "Ridge"],
    default=["Linear", "Polynomial"]
)

poly_deg = st.sidebar.number_input("Polynomial Degree", 1, 12, 3)
lasso_alpha = st.sidebar.slider("Alpha (Lasso)", 0.01, 5.0, 1.0)
ridge_alpha = st.sidebar.slider("Alpha (Ridge)", 0.01, 5.0, 1.0)
nn_size = st.sidebar.slider("NeuralNet Hidden Units", 5, 150, 40)

# ----------------------------
# Fake Data (replaceable with CSV later)
# ----------------------------
np.random.seed(0)
X_data = np.random.rand(120, 1) * 8
y_data = 4 * X_data.squeeze() + 6 + np.random.randn(120) * 2

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=0)

trained = {}
metrics = []

# ----------------------------
# Training block
# ----------------------------
if "Linear" in chosen_models:
    lin = LinearRegression().fit(X_train, y_train)
    preds = lin.predict(X_test)
    trained["Linear"] = lin
    metrics.append(["Linear", calc_rmse(y_test, preds), r2_score(y_test, preds)])

if "Polynomial" in chosen_models:
    poly = PolynomialFeatures(degree=poly_deg)
    X_tr = poly.fit_transform(X_train)
    X_te = poly.transform(X_test)
    poly_model = LinearRegression().fit(X_tr, y_train)
    preds = poly_model.predict(X_te)
    trained["Polynomial"] = (poly, poly_model)
    metrics.append([f"Polynomial(d={poly_deg})", calc_rmse(y_test, preds), r2_score(y_test, preds)])

if "RandomForest" in chosen_models:
    rf = RandomForestRegressor(n_estimators=120, random_state=0).fit(X_train, y_train)
    preds = rf.predict(X_test)
    trained["RandomForest"] = rf
    metrics.append(["RandomForest", calc_rmse(y_test, preds), r2_score(y_test, preds)])

if "NeuralNet" in chosen_models:
    nn = MLPRegressor(hidden_layer_sizes=(nn_size,), max_iter=2500, random_state=0).fit(X_train, y_train)
    preds = nn.predict(X_test)
    trained["NeuralNet"] = nn
    metrics.append(["NeuralNet", calc_rmse(y_test, preds), r2_score(y_test, preds)])

if "Lasso" in chosen_models:
    lasso = Lasso(alpha=lasso_alpha).fit(X_train, y_train)
    preds = lasso.predict(X_test)
    trained["Lasso"] = lasso
    metrics.append([f"Lasso(α={lasso_alpha})", calc_rmse(y_test, preds), r2_score(y_test, preds)])

if "Ridge" in chosen_models:
    ridge = Ridge(alpha=ridge_alpha).fit(X_train, y_train)
    preds = ridge.predict(X_test)
    trained["Ridge"] = ridge
    metrics.append([f"Ridge(α={ridge_alpha})", calc_rmse(y_test, preds), r2_score(y_test, preds)])

# ----------------------------
# Results table & bar chart
# ----------------------------
st.subheader("📊 Model Scores")
df_metrics = pd.DataFrame(metrics, columns=["Model", "RMSE", "R²"])
st.dataframe(df_metrics, use_container_width=True)

fig_bar = px.bar(df_metrics, x="Model", y="R²", color="RMSE", title="Model Performance Overview", text_auto=".2f")
st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------
# Scatter with predictions
# ----------------------------
st.subheader("📈 Predictions vs Real Data")

xx = np.linspace(0, 8, 150).reshape(-1, 1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_data.squeeze(), y=y_data, mode="markers", name="Original Data"))

if "Linear" in trained:
    fig.add_trace(go.Scatter(x=xx.flatten(), y=trained["Linear"].predict(xx), mode="lines", name="Linear"))

if "Polynomial" in trained:
    poly, poly_model = trained["Polynomial"]
    fig.add_trace(go.Scatter(x=xx.flatten(), y=poly_model.predict(poly.transform(xx)), mode="lines", name=f"Poly(d={poly_deg})"))

if "RandomForest" in trained:
    fig.add_trace(go.Scatter(x=xx.flatten(), y=trained["RandomForest"].predict(xx), mode="lines", name="RandomForest"))

if "NeuralNet" in trained:
    fig.add_trace(go.Scatter(x=xx.flatten(), y=trained["NeuralNet"].predict(xx), mode="lines", name="NeuralNet"))

if "Lasso" in trained:
    fig.add_trace(go.Scatter(x=xx.flatten(), y=trained["Lasso"].predict(xx), mode="lines", name=f"Lasso α={lasso_alpha}"))

if "Ridge" in trained:
    fig.add_trace(go.Scatter(x=xx.flatten(), y=trained["Ridge"].predict(xx), mode="lines", name=f"Ridge α={ridge_alpha}"))

st.plotly_chart(fig, use_container_width=True)

