#import packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn import datasets
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import datasets
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

#streamlit page
st.set_page_config(page_title="Evaluation Metrics", page_icon="💻")
st.markdown("# Evaluation Metrics")
st.sidebar.header("Evaluation Metrics")
st.write('This page executes the evaluation metrics and illustrate the process in real-time')

####### MAIN SECTION #######

#load data
df = pd.read_csv('product_images.csv')
true_label = pd.read_csv('true_label.csv')

#Standardization and Dimensionality Reduction with PCA
##standardization of data
scaler = StandardScaler()
df_std = scaler.fit_transform(df)


st.header('Method 1: Silhouette Score')
st.write('Silhouette Score is 0.156835')
silhouette_score = Image.open('silhouette_score.jpg')
st.image(silhouette_score)

st.header('Method 2: Davies-Bouldin Index')
st.write('Davies-Bouldin Index is 2.067655')
db_index = Image.open('db_index.jpg')
st.image(db_index)

st.header('Method 3: Adjusted Rand Index')
st.write('Adjusted Rand Index is 0.871429502950295')
