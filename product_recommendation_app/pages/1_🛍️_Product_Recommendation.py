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
st.set_page_config(page_title="Product Recommendation", page_icon="🛍️")
st.markdown("# Product Recommendation")
st.sidebar.header("Product Recommendation")
st.write('This page recommends products based on clustering algorithm')
st.write('Click the buttons to get started')

####### MAIN SECTION #######

#load data
product_df = pd.read_csv('product.csv')
true_label = pd.read_csv('true_label.csv')

# Select 10 random rows from the dataframe
product_df_random = product_df.sample(10)
# Get the pixel data for the selected rows
random_pixels = product_df_random.loc[:, 'pixel1':'pixel784'].values
# Display the selected pixels
print(random_pixels)


image_data = np.asarray(random_pixels.reshape(-1, 28, 28))

for i in range(10):
    img = Image.fromarray(np.uint8(image_data[i]))
    st.image(img, caption='Product {}'.format(i+1))
    if st.button('Product {}'.format(i+1)):
        # Get the cluster label of the random item
        cluster_label = product_df_random['cluster'].values[0]
        # Filter the dataframe to get all items belonging to that cluster
        cluster_items = product_df[product_df['cluster'] == cluster_label]
        
        # Randomly select 4 images from the same cluster
        random_images = cluster_items.sample(4)

        # Display the 4 random images
        for j in range(4):
            rec_random_pixels = random_images.loc[:, 'pixel1':'pixel784'].values
            rec_image_data = np.asarray(rec_random_pixels.reshape(-1, 28, 28))
            rec_img = Image.fromarray(np.uint8(rec_image_data[j]))
            st.image(rec_img, caption='Random image {}'.format(j+1))
                         




















