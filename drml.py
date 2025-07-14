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

st.set_page_config(page_title="Doubly Robust Machine Learning Simulation", layout="wide")

# generate random data
SAMPLES = st.number_input("Number of Samples", min_value=100, max_value=10000, value=1000, step=100)
x, y = make_classification(n_samples=SAMPLES, n_features=20, n_informative=15, random_state=42, n_classes=2)

# split data into labled and unlabeled sets
seen, unseen = sklearn.model_selection.train_test_split(
    np.arange(SAMPLES), test_size=0.2, random_state=42)
