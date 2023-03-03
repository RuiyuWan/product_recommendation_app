#import packages
import streamlit as st #streamlit for building web applications
import pandas as pd #pandas for data manipulation
import numpy as np #numpy for numerical computation
import matplotlib.pyplot as plt #matplotlib for data visualization
from PIL import Image #PIL for image manipulation
from sklearn.datasets import fetch_openml #sklearn for machine learning
from sklearn.preprocessing import StandardScaler #sklearn for machine learning
from sklearn.decomposition import PCA #sklearn for machine learning
from sklearn.cluster import KMeans #sklearn for machine learning
from sklearn.datasets import make_blobs #sklearn for machine learning
from yellowbrick.cluster import KElbowVisualizer #yellowbrick for data visualization
from sklearn import datasets #sklearn for machine learning
from sklearn.metrics import davies_bouldin_score #sklearn for machine learning
from sklearn import metrics #sklearn for machine learning
from sklearn.metrics.cluster import adjusted_rand_score #sklearn for machine learning
from sklearn import datasets #sklearn for machine learning
from sklearn.metrics import silhouette_score #sklearn for machine learning
from yellowbrick.cluster import SilhouetteVisualizer #yellowbrick for data visualization

#streamlit page
st.set_page_config(page_title="Product Recommendation", page_icon="🛍️") #set page title and icon
st.markdown("# Product Recommendation") #set page title
st.sidebar.header("Product Recommendation") #set sidebar title
st.write('This page recommends products based on clustering algorithm') #set page description
st.write('Click the buttons to get started') #set page description

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


image_data = np.asarray(random_pixels.reshape(-1, 28, 28)) #reshape the data

for i in range(10): #loop through the images
    img = Image.fromarray(np.uint8(image_data[i])) #convert the data to image
    st.image(img, caption='Product {}'.format(i+1)) #display the image
    if st.button('Product {}'.format(i+1)): #display the button
        # Get the cluster label of the random item
        cluster_label = product_df_random['cluster'].values[0]
        # Filter the dataframe to get all items belonging to that cluster
        cluster_items = product_df[product_df['cluster'] == cluster_label]
        
        # Randomly select 4 images from the same cluster
        random_images = cluster_items.sample(4)

        # Display the 4 random images
        for j in range(4): #loop through the images
            rec_random_pixels = random_images.loc[:, 'pixel1':'pixel784'].values #get the pixel data
            rec_image_data = np.asarray(rec_random_pixels.reshape(-1, 28, 28)) #reshape the data
            rec_img = Image.fromarray(np.uint8(rec_image_data[j])) #convert the data to image
            st.image(rec_img, caption='Random image {}'.format(j+1)) #display the image
