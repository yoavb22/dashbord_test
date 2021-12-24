"""
Geometry of information - Ex.3

12/11/2021

Neria Rivlin and Yoav Blonder
"""

# Question 4

# Imports
import numpy as np
import pandas as pd
from scipy import io
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import streamlit as st
import matplotlib.pyplot as plt
#
# import sys, asyncio
# import cv2

# if sys.platform == "win32" and (3, 6, 0) <= sys.version_info < (3, 9, 0):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# n = 100
# p_vec = [0.7, 0.7, 1, 0]
# q_vec = [0, 0.6, 1, 0.7]
# # streamlit run EX_3.py
#
# # generate stochastic block model matrix
# iu1 = np.triu_indices(n, 1)
# fig, ax = plt.subplots(4, 4, figsize=(10, 10))
# for i in range(len(q_vec)):
#     W_AA = np.zeros((n, n))
#     W_BA = np.zeros((n, n))
#     W_BB = np.zeros((n, n))
#     q = q_vec[i]
#     p = p_vec[i]
#
#     # set random binary edges for upper triangle of matrix
#     W_AA[iu1] = np.random.binomial(1, p, int(n * (n - 1) / 2))
#     W_BB[iu1] = np.random.binomial(1, p, int(n * (n - 1) / 2))
#     W_BA = np.random.binomial(1, q, size=(n, n))
#     W = np.concatenate((np.concatenate((W_AA, W_BA.T), axis=1), np.concatenate((W_BA, W_BB), axis=1)), axis=0)
#
#     # Graph is unweighted - make matrix symmetric
#     W = W + W.transpose()
#     L = np.diag(np.sum(W, axis=1)) - W
#
#     # compute eigenvalues and eigenvectors of Laplacian
#     [D, V] = np.linalg.eigh(L)
#
#     # show matrix, Fiedler vector and eigenvalues
#     ax[i, 0].plot(V[:, 2 * n - 1])
#     ax[i, 0].title.set_text("eigenvectors 1")
#     ax[i, 1].plot(V[:, 2 * n - 2])
#     ax[i, 1].title.set_text("eigenvectors 2")
#     ax[i, 2].plot(V[:, 2 * n - 3])
#     ax[i, 2].title.set_text("eigenvectors 3")
#     ax[i, 3].plot(np.abs(D[(n - 3):n]), 'bo')
#     ax[i, 3].title.set_text('q=' + str(q) + " p=" + str(p))
#     fig.tight_layout()
# fig.show()
# Q6
st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://en.wikipedia.org/wiki/Pikachu',
        'Report a bug': "https://www.linkedin.com/in/blonder-yoav-554a00211/",
        'About': "# Neria yach ze dey magniv"
    }
)

path = "C:\\Users\\Yoav\\Documents\\university\\year4semester1\\geomety of data\\ex\\ex2\\Data6.mat"
# data_6 = io.loadmat(path)


def plot_data(df):
    rows, cols = 3, 2
    fig_1, axs = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            axs[row, col].scatter(df["XX"][0][((2 * row) + col)][:, 0], df["XX"][0][((2 * row) + col)][:, 1])
    # fig_1.set_size_inches(10, 10)
    fig_1.tight_layout()
    return fig_1


# st.pyplot(plot_data(data_6))
# smile = data_6["XX"][0][2]
# line = data_6["XX"][0][4]


# smile

def plot_kmeans_clustering(df):
    rows, cols = 3, 2
    fig_1, axs = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            clustering_1 = SpectralClustering(n_clusters=((2 * row) + col + 2)).fit(df)
            axs[row, col].scatter(df[:, 0], df[:, 1], c=clustering_1.labels_)
            axs[row, col].set_title(str((2 * row) + col + 2) + " components")
    # fig_1.set_size_inches(10, 10)
    fig_1.suptitle("k-means, spectral clustering")
    fig_1.tight_layout()
    return fig_1


#
# st.pyplot(plot_kmeans_clustering(smile))
# st.pyplot(plot_kmeans_clustering(line))

def plot_kmeans_plus_plus_clustering(df):
    rows, cols = 3, 2
    fig_1, axs = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            clustering_1 = KMeans(n_clusters=((2 * row) + col + 2), init='k-means++').fit(df)
            axs[row, col].scatter(df[:, 0], df[:, 1], c=clustering_1.labels_)
            axs[row, col].set_title(str((2 * row) + col + 2) + " components")
    # fig_1.set_size_inches(10, 10)
    fig_1.suptitle("k-means, self-tuning spectral clustering")
    fig_1.tight_layout()
    return (fig_1)


# plot_kmeans_plus_plus_clustering(smile)
# plot_kmeans_plus_plus_clustering(line)
def main():
    uploaded_file_1 = st.file_uploader(path)
    data_6 = io.loadmat(uploaded_file_1)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("k-means, spectral clustering smile plot")
        st.pyplot(plot_kmeans_clustering(smile))
        with st.expander("See explanation"):
            st.write("""
                has you can see in the k-means, spectral clustering  with k = 4/5 the eyes not component      
            """)

    with col2:
        # # st.subheader("line plot")
        # # st.pyplot(plot_kmeans_clustering(line))
        # st.subheader("k-means, self-tuning spectral clustering smile plot")
        # st.pyplot(plot_kmeans_plus_plus_clustering(smile))
        # with st.expander("See explanation"):
        #     st.write("""
        #         has you can see in the k-means, self-tuning spectral clustering
        #           with k = 4/5 the eyes component
        #           """)
        st.subheader("data frame")
        data_number = st.slider('which data you want?', 0, 5)
        cul_number = st.selectbox("choose cul", pd.DataFrame(smile).keys())
        st.dataframe(data=pd.DataFrame(data_6["XX"][0][data_number])[cul_number])

    # with col3:
    #     title = st.text_input('choose data frame smile or line', '')
    #     st.write('The current movie title is', title)

    with col3:
        st.subheader("Steph Yuval & Ariella")
        video_file = open('C:\\Users\\Yoav\\Downloads\\IMG_6713.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
main()
