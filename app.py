import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.title("YouTube Video Performance Predictor Dashboard")

st.image("header.png")  # optional

st.markdown("""
Enter a YouTube video URL below.  
The app will show the **current vs predicted views and likes**.
""")

# Load CSV
df = pd.read_csv("youtube_data_deploy.csv")

# User input
video_url = st.text_input("YouTube Video URL")

if st.button("Predict"):
    if video_url:
        # Extract video ID
        video_id = video_url.split("v=")[-1].split("&")[0]

        # Find video in CSV
        video_row = df[df['video_id'] == video_id]

        if video_row.empty:
            st.error("Video not found in the CSV.")
        else:
            view_count = int(video_row['view_count'].values[0])
            likes = int(video_row['likes'].values[0])

            st.write(f"**Current Views:** {view_count}")
            st.write(f"**Current Likes:** {likes}")

            # Load models
            model_viewcount = joblib.load("boosted_tree_model_viewcount.pkl")
            model_likes = joblib.load("boosted_tree_model_likes.pkl")

            # Predict
            features = pd.DataFrame([[view_count, likes]], columns=["view_count", "likes"])
            predicted_views = model_viewcount.predict(features)[0]
            predicted_likes = model_likes.predict(features)[0]

            st.success("Predictions:")
            st.write(f"Predicted Views: {int(predicted_views)}")
            st.write(f"Predicted Likes: {int(predicted_likes)}")

            # --- Dashboard ---
            fig = go.Figure(data=[
                go.Bar(name='Current', x=['Views', 'Likes'], y=[view_count, likes]),
                go.Bar(name='Predicted', x=['Views', 'Likes'], y=[predicted_views, predicted_likes])
            ])
            fig.update_layout(title_text='Current vs Predicted Video Performance',
                              barmode='group', yaxis_title='Count')
            st.plotly_chart(fig)
    else:
        st.error("Please enter a YouTube URL.")
