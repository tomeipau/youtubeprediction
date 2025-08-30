import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

st.set_page_config(page_title="YouTube Video Analysis", layout="wide")

# Header image
if os.path.exists("header.png"):
    st.image("header.png", use_container_width=True)

st.title("YouTube Video Performance Dashboard")

# Load dataset safely
if os.path.exists("youtube_data_deploy.csv"):
    df = pd.read_csv("youtube_data_deploy.csv")
else:
    st.error("Dataset 'youtube_data_deploy.csv' not found. Please upload it.")
    st.stop()

# Sidebar for filtering
st.sidebar.header("Filter Options")
channel_list = ["All"] + sorted(df['channelTitle'].dropna().unique().tolist())
selected_channel = st.sidebar.selectbox("Select Channel", channel_list)

if selected_channel != "All":
    df = df[df['channelTitle'] == selected_channel]

# Video URL filter
video_url_input = st.sidebar.text_input("Enter YouTube Video URL (optional):")
if video_url_input:
    video_id = video_url_input.split("v=")[-1].split("&")[0]
    df = df[df['video_id'] == video_id]

# ==============================
# 1. Top Performing Videos
# ==============================
st.subheader("Top Performing Videos by Views")
top_videos = df.sort_values(by='view_count', ascending=False).head(10)
st.dataframe(top_videos[['video_id', 'channelTitle', 'view_count', 'likes']])

# ==============================
# 2. Category-Level Performance
# ==============================
st.subheader("Category Performance")
category_performance = df.groupby('category_encoded')[['view_count', 'likes']].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(x='category_encoded', y='view_count', data=category_performance, ax=ax)
ax.set_title("Average Views by Category")
st.pyplot(fig)

# ==============================
# 3. Engagement Distribution
# ==============================
st.subheader("Engagement Distribution")
fig, ax = plt.subplots()
sns.histplot(df['likes_per_view'], bins=30, kde=True, ax=ax)
ax.set_title("Likes per View Distribution")
st.pyplot(fig)

# ==============================
# 4. Correlation Heatmap
# ==============================
st.subheader("Feature Correlation Heatmap")
corr = df[['view_count', 'likes', 'comment_count', 'title_score', 'tags_score']].corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# ==============================
# 5. Predictive Modeling
# ==============================
st.subheader("Predict Video Performance")

# Load Models
try:
    with open("boosted_tree_model_viewcount.pkl", "rb") as f:
        model_view = pickle.load(f)
    with open("boosted_tree_model_likes.pkl", "rb") as f:
        model_likes = pickle.load(f)

    st.write("Enter video features:")
    title_score = st.number_input("Title Score", 0.0, 1.0, 0.5)
    tags_score = st.number_input("Tags Score", 0.0, 1.0, 0.5)
    comment_count = st.number_input("Comment Count", 0, 1000000, 100)
    category_encoded = st.number_input("Category Encoded", 0, 50, 5)

    if st.button("Predict"):
        features = [[title_score, tags_score, comment_count, category_encoded]]
        predicted_views = model_view.predict(features)[0]
        predicted_likes = model_likes.predict(features)[0]
        st.success(f"Predicted Views: {int(predicted_views)}")
        st.success(f"Predicted Likes: {int(predicted_likes)}")

except FileNotFoundError:
    st.warning("Prediction models not found. Please upload .pkl files.")
