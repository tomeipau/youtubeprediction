import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page layout
st.set_page_config(layout="wide")

# Load models
model_viewcount = joblib.load("boosted_tree_model_viewcount.pkl")
model_likes = joblib.load("boosted_tree_model_likes.pkl")

# Load dataset
df = pd.read_csv("youtube_data_deploy.csv")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section",
    ["Introduction", "Analysis Dashboard", "Prediction"]
)

# ------------------------
# INTRODUCTION
# ------------------------
def show_introduction():
    st.title("YouTube Video Performance Predictor")
    st.image("header.png")
    st.markdown("""
    This app predicts future views and likes of YouTube videos based on current metrics,
    sentiment scores (via Gemini LLM), and metadata encodings.
    """)

# ------------------------
# ANALYSIS DASHBOARD
# ------------------------
def show_analysis():
    st.title("Dataset Dashboard")

    # --- Video Filter ---
    st.subheader("Filter by YouTube Video Link")
    video_url = st.text_input("Paste YouTube Video URL (optional)")

    if video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]
        filtered_df = df[df['video_id'] == video_id]
        if filtered_df.empty:
            st.warning("Video not found. Showing full dataset instead.")
            filtered_df = df
    else:
        filtered_df = df

    tab1, tab2, tab3 = st.tabs(["Overview", "Engagement Trends", "Sentiment Analysis"])

    # --- TAB 1: Overview ---
    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(filtered_df.head(10))
        st.markdown("""
    Data record is captured on different days, and each row consist of different sentiment scorings
    """)
        #st.write(f"Total Records: {len(filtered_df)}")

        # Performance Overview
st.subheader("Performance Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Views", f"{df['view_count'].sum():,.0f}")
col2.metric("Total Likes", f"{df['likes'].sum():,.0f}")
col3.metric("Total Comments", f"{df['comment_count'].sum():,.0f}")

        # Correlation Heatmap
        corr = filtered_df[[
            "view_count", "likes", "dislikes", "comment_count",
            "views_per_day", "likes_per_view",
            "title_score", "description_score", "tags_score"
        ]].corr()

        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto",
            title="Correlation Heatmap of Key Features"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 2: Engagement Trends ---
    with tab2:
        st.subheader("Engagement Trends Over Time")

        if video_url and not filtered_df.empty:
            fig_views = px.line(
                filtered_df, x="days_to_trend", y="view_count",
                title="View Count vs Days to Trend"
            )
            st.plotly_chart(fig_views, use_container_width=True)

            fig_likes = px.line(
                filtered_df, x="days_to_trend", y="likes",
                title="Likes vs Days to Trend"
            )
            st.plotly_chart(fig_likes, use_container_width=True)
        else:
            st.info("Please paste a YouTube video link above to view engagement trends.")


    # --- TAB 3: Sentiment Analysis ---
    with tab3:
        st.subheader("Sentiment Score Distribution")
        sentiment_cols = ["title_score", "description_score", "tags_score"]
        for col in sentiment_cols:
            fig_sentiment = px.histogram(
                filtered_df, x=col, nbins=30,
                title=f"Distribution of {col}"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

# ------------------------
# PREDICTION
# ------------------------
def show_prediction():
    st.header("Predict YouTube Video Performance")
    video_url = st.text_input("Enter YouTube Video URL")

    if video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]
        video_row = df[df['video_id'] == video_id]

        if video_row.empty:
            st.error("Video not found in dataset.")
        else:
            view_count = int(video_row['view_count'].values[0])
            likes = int(video_row['likes'].values[0])

            st.write(f"**Current Views:** {view_count}")
            st.write(f"**Current Likes:** {likes}")

            features = pd.DataFrame([[view_count, likes]], columns=["view_count", "likes"])
            predicted_views = model_viewcount.predict(features)[0]
            predicted_likes = model_likes.predict(features)[0]

            st.success("Predicted Performance")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Views", int(predicted_views))
            col2.metric("Predicted Likes", int(predicted_likes))

            fig = go.Figure(data=[
                go.Bar(name='Current', x=['Views', 'Likes'], y=[view_count, likes]),
                go.Bar(name='Predicted', x=['Views', 'Likes'], y=[predicted_views, predicted_likes])
            ])
            fig.update_layout(
                title_text='Current vs Predicted Video Performance',
                barmode='group', yaxis_title='Count'
            )
            st.plotly_chart(fig)

# ------------------------
# PAGE SELECTION
# ------------------------
if section == "Introduction":
    show_introduction()
elif section == "Analysis Dashboard":
    show_analysis()
elif section == "Prediction":
    show_prediction()
