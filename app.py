import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Set page layout to wide
st.set_page_config(layout="wide")

# Load models
model_viewcount = joblib.load("boosted_tree_model_viewcount.pkl")
model_likes = joblib.load("boosted_tree_model_likes.pkl")

# Load CSV
df = pd.read_csv("youtube_data_deploy.csv")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a section", ["Introduction", "Prediction", "Analysis"])

# Introduction Section
def show_introduction():
    st.title("YouTube Video Performance Predictor")
    st.image("header.png")
    st.markdown("""
    This app predicts the future views and likes of a YouTube video based on its current statistics.
    Enter a YouTube video URL to get predictions.
    """)

# Analysis Section
def show_analysis():
    st.header("Data Analysis")
    st.write("Here you can add analysis of the dataset, such as trends, distributions, or correlations.")
    
# Prediction Section
def show_prediction():
    st.header("Enter YouTube Video URL")
    video_url = st.text_input("YouTube Video URL")

    if video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]
        video_row = df[df['video_id'] == video_id]

        if video_row.empty:
            st.error("Video not found in the dataset.")
        else:
            view_count = int(video_row['view_count'].values[0])
            likes = int(video_row['likes'].values[0])

            st.write(f"**Current Views:** {view_count}")
            st.write(f"**Current Likes:** {likes}")

            features = pd.DataFrame([[view_count, likes]], columns=["view_count", "likes"])
            predicted_views = model_viewcount.predict(features)[0]
            predicted_likes = model_likes.predict(features)[0]

            st.success("Predictions:")
            st.write(f"Predicted Views: {int(predicted_views)}")
            st.write(f"Predicted Likes: {int(predicted_likes)}")

            # Plotting
            fig = go.Figure(data=[
                go.Bar(name='Current', x=['Views', 'Likes'], y=[view_count, likes]),
                go.Bar(name='Predicted', x=['Views', 'Likes'], y=[predicted_views, predicted_likes])
            ])
            fig.update_layout(title_text='Current vs Predicted Video Performance', barmode='group', yaxis_title='Count')
            st.plotly_chart(fig)

# Display the selected section
if section == "Introduction":
    show_introduction()
elif section == "Analysis":
    show_analysis()
elif section == "Prediction":
    show_prediction()
