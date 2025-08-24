import streamlit as st
import pandas as pd
import joblib

st.title("YouTube Video Performance Predictor")

st.image("header.png")  # optional header

st.markdown("""
Enter a YouTube video URL below.  
For this demo, the app uses realistic numbers for views and likes.
""")

# User input for video URL
video_url = st.text_input("YouTube Video URL")

# Demo numbers matching real video
demo_views = 3079371
demo_likes = 56000

if st.button("Predict"):
    if video_url:
        # Extract video ID (simple parsing)
        video_id = video_url.split("v=")[-1].split("&")[0]

        # Load local models
        model_viewcount = joblib.load("boosted_tree_model_viewcount.pkl")
        model_likes = joblib.load("boosted_tree_model_likes.pkl")

        # Create features for prediction
        features = pd.DataFrame([[demo_views, demo_likes]], columns=["view_count", "likes"])

        # Predict
        predicted_views = model_viewcount.predict(features)[0]
        predicted_likes = model_likes.predict(features)[0]

        st.success(f"**Video ID:** {video_id}")
        st.write(f"Predicted Views: {predicted_views}")
        st.write(f"Predicted Likes: {predicted_likes}")
    else:
        st.error("Please enter a YouTube URL.")
