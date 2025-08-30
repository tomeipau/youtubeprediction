import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Load models
model_viewcount = joblib.load("boosted_tree_model_viewcount.pkl")
model_likes = joblib.load("boosted_tree_model_likes.pkl")

# App Title
st.title("YouTube Video Performance Dashboard")

# --- SECTION 1: URL Input & Basic Video Info ---
st.header("1. Enter YouTube Video URL")
youtube_url = st.text_input("Paste a YouTube video URL:")

if youtube_url:
    st.write(f"Video Link: {youtube_url}")
    video_id = youtube_url.split("v=")[-1]
    video_data = df[df["video_id"] == video_id]
    
    if not video_data.empty:
        st.subheader("Basic Video Data")
        st.dataframe(video_data)
    else:
        st.warning("No data found for this video.")

# --- SECTION 2: Prediction Section ---
st.header("2. Predict Performance")
st.write("Prediction uses trained Boosted Tree models.")

if youtube_url and not video_data.empty:
    features = video_data.drop(columns=["video_id", "youtube_link"])
    pred_views = model_viewcount.predict(features)[0]
    pred_likes = model_likes.predict(features)[0]

    st.metric("Predicted Views", f"{int(pred_views):,}")
    st.metric("Predicted Likes", f"{int(pred_likes):,}")

# --- SECTION 3: Correlation Heatmap ---
st.header("3. Correlation Heatmap")
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=False, cmap="coolwarm")
st.pyplot(plt)

# --- SECTION 4: Top Performing Videos ---
st.header("4. Top Performing Videos")
top_videos = df.groupby("video_id")["view_count"].max().sort_values(ascending=False).head(10)
st.write(top_videos)
