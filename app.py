import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Page Config
st.set_page_config(page_title="YouTube Video Analysis", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("youtube_data_deploy.csv")

df = load_data()

# Sidebar Filters
st.sidebar.image("header.png", use_container_width=True)
st.sidebar.title("Filters")

# Replace channel filter with video_id filter
video_list = ["All"] + sorted(df['video_id'].dropna().unique().tolist())
selected_video = st.sidebar.selectbox("Select Video", video_list)

if selected_video != "All":
    df = df[df['video_id'] == selected_video]

# Title
st.title("YouTube Video Performance Dashboard")

# --- SECTION 1: Performance Overview ---
st.subheader("Performance Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Views", f"{df['view_count'].sum():,.0f}")
col2.metric("Total Likes", f"{df['likes'].sum():,.0f}")
col3.metric("Total Comments", f"{df['comment_count'].sum():,.0f}")

# --- SECTION 2: Engagement Metrics ---
st.subheader("Engagement Metrics")
fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(data=df, x='view_count', y='likes', ax=ax)
ax.set_title("Likes vs View Count")
st.pyplot(fig)

# --- SECTION 3: Content Quality Analysis ---
st.subheader("Content Quality Scores")
st.write(df[['title_score', 'description_score', 'tags_score']].describe())

# --- SECTION 4: Category Analysis ---
st.subheader("Category Analysis")
if 'category_encoded' in df.columns:
    cat_fig, cat_ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='category_encoded', data=df, ax=cat_ax)
    cat_ax.set_title("Distribution of Categories")
    st.pyplot(cat_fig)

# --- SECTION 5: Temporal Analysis ---
st.subheader("Temporal Trends")
if 'month_encoded' in df.columns:
    temp_fig, temp_ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x='month_encoded', y='view_count', data=df, ax=temp_ax)
    temp_ax.set_title("Views by Month")
    st.pyplot(temp_fig)

# --- SECTION 6: Predictive Modelling ---
st.subheader("Predictive Modelling")

# Load Pre-Trained Models
with open("boosted_tree_model_likes.pkl", "rb") as f:
    model_likes = pickle.load(f)
with open("boosted_tree_model_viewcount.pkl", "rb") as f:
    model_views = pickle.load(f)

st.markdown("### Predict Views and Likes")
input_features = df.drop(columns=['video_id', 'youtube_link'], errors='ignore').mean().to_frame().T
pred_views = model_views.predict(input_features)[0]
pred_likes = model_likes.predict(input_features)[0]

st.write(f"**Predicted Views:** {pred_views:,.0f}")
st.write(f"**Predicted Likes:** {pred_likes:,.0f}")

# --- SECTION 7: Video Links ---
st.subheader("Watch Video")
if 'youtube_link' in df.columns:
    for link in df['youtube_link'].unique():
        st.markdown(f"[Watch on YouTube]({link})")

