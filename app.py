import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  

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
    ["Background", "Overview", "Prediction"]
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
    st.title("Analysis & Video Performance Dashboard")

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

    tab1, tab2, tab3 = st.tabs(["Video Overview", "Project Overview", "Sentiment Analysis"])

    # --- TAB 1: Overview ---
    with tab1:
        # Performance Overview
        st.subheader("Performance Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Views", f"{filtered_df['view_count'].max():,.0f}")
        col2.metric("Total Likes", f"{filtered_df['likes'].max():,.0f}")
        col3.metric("Total Comments", f"{filtered_df['comment_count'].max():,.0f}")

                #--EDA
        st.subheader("Exploratory Data Analysis")
        st.image("EDA.png")
        st.markdown("""
        This app predicts future views and likes of YouTube videos based on current metrics,
        sentiment scores (via Gemini LLM), and metadata encodings.
        """)

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

    # --- TAB 2: Project Overview ---
    with tab2:
        #st.subheader("Project Overview")
        
        #--Dataset Overview
        st.subheader("Dataset Overview")
        st.dataframe(filtered_df.head(5))
        st.markdown("""
    Data is extracted from Youtube API V3 and downloaded from https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset?select=US_youtube_trending_data.csv. 
    Data consists of records of nideo title, channel title, publish time, tags, views, likes and dislikes, description, and comment count that is extracted daily
    """)

         #--Processed Dataset (Encodings & LLM Embedded) Overview
        st.subheader("Processed Dataset (LLM Embedded) Overview")
        st.dataframe(filtered_df.head(5))
        st.markdown("""
    Dataset was processed in Bigquery for LLM sentiment scoring for the columns title, description and tags. Encodings were also done for non-numerical values.
    """)
        #st.write(f"Total Records: {len(filtered_df)}")

        #--EDA
        st.subheader("Exploratory Data Analysis")
        st.image("EDA.png")
        st.markdown("""
        This app predicts future views and likes of YouTube videos based on current metrics,
        sentiment scores (via Gemini LLM), and metadata encodings.
        """)

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
        st.markdown("""
        ### Sentiment Scoring Overview
        This dashboard uses **Google's Gemini LLM** to analyze text data (video titles, descriptions, and tags).  
        Each piece of text is scored based on **sentiment polarity** and **confidence level**, which are derived from the model's predictions.  
        
        - **Title Score**: Sentiment derived from video title.  
        - **Description Score**: Sentiment from video description.  
        - **Tags Score**: Sentiment based on video tags.  
        
        **Score Meaning**:  
        - Scores range from `-1` to `1`.  
        - `-1` indicates a highly negative sentiment.  
        - `0` represents a neutral sentiment.  
        - `1` indicates a highly positive sentiment.  
        
        Gemini provides advanced contextual understanding, ensuring more accurate sentiment representation.
        """)
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

            features = pd.DataFrame([[view_count, likes]], columns=["view_count", "likes"])
            predicted_views = model_viewcount.predict(features)[0]
            predicted_likes = model_likes.predict(features)[0]

            st.success("Predicted Performance")
            col1, col2 = st.columns(2)
            
            #--Predicted metrics
            col1.metric("Predicted Views", f"{predicted_views:,.0f}")
            col2.metric("Predicted Likes", f"{predicted_likes:,.0f}")
            
            #--Smaller text for current values under metrics
            col1.markdown(f"<span style='font-size:12px; color:gray'>Current: {view_count:,.0f}</span>", unsafe_allow_html=True)
            col2.markdown(f"<span style='font-size:12px; color:gray'>Current: {likes:,.0f}</span>", unsafe_allow_html=True)


            # --- Separated Subplots ---
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Views", "Likes"))

            # Views Bar
            fig.add_trace(
                go.Bar(
                    name='Current Views',
                    x=['Current'], y=[view_count],
                    text=[f"{view_count:,.0f}"],
                    textposition='auto'
                ), row=1, col=1
            )
            fig.add_trace(
                go.Bar(
                    name='Predicted Views',
                    x=['Predicted'], y=[predicted_views],
                    text=[f"{predicted_views:,.0f}"],
                    textposition='auto'
                ), row=1, col=1
            )

            # Likes Bar
            fig.add_trace(
                go.Bar(
                    name='Current Likes',
                    x=['Current'], y=[likes],
                    text=[f"{likes:,.0f}"],
                    textposition='auto'
                ), row=1, col=2
            )
            fig.add_trace(
                go.Bar(
                    name='Predicted Likes',
                    x=['Predicted'], y=[predicted_likes],
                    text=[f"{predicted_likes:,.0f}"],
                    textposition='auto'
                ), row=1, col=2
            )

            fig.update_layout(
                title_text='Current vs Predicted Video Performance',
                showlegend=True,
                height=500,
                barmode='group',
                yaxis_title='Count'
            )

            st.plotly_chart(fig, use_container_width=True)

# ------------------------
# PAGE SELECTION
# ------------------------
if section == "Background":
    show_introduction()
elif section == "Overview":
    show_analysis()
elif section == "Prediction":
    show_prediction()

