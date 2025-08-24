import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("YouTube Video Performance Predictor")

# Optional header
st.image("header.png")

st.markdown("""
Enter a YouTube video URL below.  
Since we don't have access to the API yet, please also input approximate `view_count` and `likes`.
""")

# User input
video_url = st.text_input("YouTube Video URL")
view_count_input = st.number_input("Approximate current views", min_value=0, value=1000)
likes_input = st.number_input("Approximate current likes", min_value=0, value=100)

if st.button("Predict"):
    if video_url:
        # Extract video ID from URL (simple parsing)
        video_id = video_url.split("v=")[-1].split("&")[0]

        # Load local models
        model_viewcount = joblib.load("boosted_tree_model_viewcount.pkl")
        model_likes = joblib.load("boosted_tree_model_likes.pkl")

        # Create feature dataframe
        features = pd.DataFrame([[view_count_input, likes_input]], columns=["view_count", "likes"])

        # Predict
        predicted_views = model_viewcount.predict(features)[0]
        predicted_likes = model_likes.predict(features)[0]

        st.success(f"**Video ID:** {video_id}")
        st.write(f"Predicted Views: {int(predicted_views)}")
        st.write(f"Predicted Likes: {int(predicted_likes)}")
    else:
        st.error("Please enter a YouTube URL.")
