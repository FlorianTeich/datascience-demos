# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn",
#     "streamlit",
# ]
# ///
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Federated Learning Simulation", layout="wide")
st.header("Federated Learning Parameters")
SAMPLES = st.number_input("Number of Samples", min_value=100, max_value=10000, value=1000, step=100)
CLIENTS = st.number_input("Number of Clients", min_value=2, max_value=10, value=4, step=1)
EPOCHS = st.number_input("Number of Epochs", min_value=1, max_value=100, value=50, step=1)

N_FEATURES = st.number_input("Number of Features", min_value=1, max_value=100, value=20, step=1)
N_INFORMATIVE = st.number_input("Number of Informative Features", min_value=1, 
                                max_value=N_FEATURES, value=int(N_FEATURES*0.75), step=1)
N_CLASSES = st.number_input("Number of Classes", min_value=2, max_value=10, value=2, step=1)
N_CLUSTERS_PER_CLASS = st.number_input("Number of Clusters per Class", min_value=1, max_value=10, value=2, step=1)
FLIP_Y = st.slider("Fraction of Samples to Flip", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
CLASS_SEP = st.slider("Class Separation", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
ITERS_PER_CLIENT = st.number_input("Iterations per Client per Epoch", min_value=1, max_value=1000, value=100, step=1)

# generate random data
x, y = make_classification(
    n_samples=SAMPLES,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    random_state=42,
    n_classes=N_CLASSES,
    n_clusters_per_class=N_CLUSTERS_PER_CLASS,
    flip_y=FLIP_Y,
    class_sep=CLASS_SEP,
)

# split data into clients
random_assignments = np.random.randint(0, CLIENTS, size=SAMPLES)
client_data_x = [x[random_assignments == i] for i in range(CLIENTS)]
client_data_y = [y[random_assignments == i] for i in range(CLIENTS)]
# split data into train and test sets for each client
client_data_x_train = [client_data_x[i][:int(len(client_data_x[i]) * 0.8)] for i in range(CLIENTS)]
client_data_x_test = [client_data_x[i][int(len(client_data_x[i]) * 0.8):] for i in range(CLIENTS)]
client_data_y_train = [client_data_y[i][:int(len(client_data_y[i]) * 0.8)] for i in range(CLIENTS)]
client_data_y_test = [client_data_y[i][int(len(client_data_y[i]) * 0.8):] for i in range(CLIENTS)]

client_models = [MLPClassifier(hidden_layer_sizes=(10,), max_iter=ITERS_PER_CLIENT, warm_start=True) for _ in range(CLIENTS)]

# Isolated Training
isolated_aurocs = []
isolated_models = [MLPClassifier(hidden_layer_sizes=(10,), max_iter=ITERS_PER_CLIENT, warm_start=True) for _ in range(CLIENTS)]
auroc_history = []
for epoch in range(1000):
    for i in range(CLIENTS):
        isolated_models[i].fit(client_data_x_train[i], client_data_y_train[i])
        preds = isolated_models[i].predict(client_data_x_test[i])  # This line is just to trigger the warm start
        isolated_aurocs.append(roc_auc_score(client_data_y_test[i], preds))
    mean_aurocs = np.mean(isolated_aurocs)
    auroc_history.append(mean_aurocs)
    print(f"Epoch {epoch + 1}, Isolated Client AUROCs: {mean_aurocs:.4f}")
    if epoch > 0 and abs(auroc_history[-1] - auroc_history[-2]) < 0.0001:
        print("Convergence reached, stopping training.")
        break

# Simulate global pooled training
pooled_aurocs = []
pooled_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=ITERS_PER_CLIENT, warm_start=True)
auroc_history = []
train_data_x = np.concatenate(client_data_x_train)
train_data_y = np.concatenate(client_data_y_train)
test_data_x = np.concatenate(client_data_x_test)
test_data_y = np.concatenate(client_data_y_test)
for epoch in range(1000):
    pooled_model.fit(train_data_x, train_data_y)
    preds = pooled_model.predict(test_data_x)  # This line is just to trigger the warm start
    pooled_aurocs.append(roc_auc_score(test_data_y, preds))
    mean_aurocs = np.mean(pooled_aurocs)
    auroc_history.append(mean_aurocs)
    print(f"Epoch {epoch + 1}, Single Global AUROCs: {mean_aurocs:.4f}")
    if epoch > 0 and abs(auroc_history[-1] - auroc_history[-2]) < 0.0001:
        print("Convergence reached, stopping training.")
        break

# Federated Learning Simulation
auroc_history = []
for epoch in range(EPOCHS):
    client_aurocs = []
    for i in range(CLIENTS):
        client_models[i].fit(client_data_x_train[i], client_data_y_train[i])

        preds = client_models[i].predict(client_data_x_test[i])  # This line is just to trigger the warm start
        client_aurocs.append(roc_auc_score(client_data_y_test[i], preds))

    # Aggregate model parameters
    for layer in range(len(client_models[0].coefs_)):
        global_weights = np.mean([model.coefs_[layer] for model in client_models], axis=0)
        global_intercepts = np.mean([model.intercepts_[layer] for model in client_models], axis=0)
        for model in client_models:
            model.coefs_[layer] = global_weights
            model.intercepts_[layer] = global_intercepts
    
    mean_aurocs = np.mean(client_aurocs)
    auroc_history.append(mean_aurocs)
    print(f"Epoch {epoch + 1}, Client AUROCs: {mean_aurocs:.4f}")
    if epoch > 0 and abs(auroc_history[-1] - auroc_history[-2]) < 0.0001:
        print("Convergence reached, stopping training.")
        break

# Plotting the results
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(1, len(auroc_history) + 1)),
    y=auroc_history,
    mode='lines+markers',
    name='Federated Learning AUROC',
    line=dict(color='blue', width=2)
))
# Add isolated AUROC for comparison
isolated_aurocs_mean = isolated_aurocs[-1]
fig.add_trace(go.Scatter(
    x=list(range(1, len(auroc_history) + 1)),
    y=[isolated_aurocs_mean] * len(auroc_history),
    mode='lines',
    name='Isolated AUROC',
    line=dict(color='red', width=2, dash='dash')
))
# Add pooled AUROC for comparison
pooled_aurocs_mean = pooled_aurocs[-1]
fig.add_trace(go.Scatter(
    x=list(range(1, len(auroc_history) + 1)),
    y=[pooled_aurocs_mean] * len(auroc_history),
    mode='lines',
    name='Pooled AUROC',
    line=dict(color='green', width=2, dash='dash')
))
fig.update_layout(
    title='Federated Learning AUROC Over Epochs',
    xaxis_title='Epochs',
    yaxis_title='AUROC',
    template='plotly_white'
)
st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap

        streamlit.web.bootstrap.run(__file__, False, [], {})
