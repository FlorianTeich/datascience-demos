import streamlit as st
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


SAMPLES = 1000
CLIENTS = 4
EPOCHS = 50

# generate random data
x, y = make_classification(
    n_samples=SAMPLES,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
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

client_models = [MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, warm_start=True) for _ in range(CLIENTS)]

# Isolated Training
isolated_aurocs = []
isolated_models = [MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, warm_start=True) for _ in range(CLIENTS)]
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
