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
df_raw = pd.read_csv("sampled_videos.csv")

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
    # st.title("YouTube Video Performance Predictor")
    st.image("header.png")

    # Introduction
    st.subheader("Introduction")
    st.markdown("""
    Machine learning models and big data solutions have demonstrated significant potential in driving content optimization 
    and audience engagement on platforms like **YouTube**.  

    However, existing research often focuses on **traditional methods** such as recommendation systems or basic prediction 
    models, which may not fully capture deeper insights into the real-world dynamics of content creation.  

    Moreover, studies often lack discussions on integrating **machine learning within big data contexts**.  

    Hence, this study aims to bridge this gap by incorporating **sentiment analysis powered by Large Language Models (LLMs)**, 
    alongside other performance indicators within a **cloud-based framework**, to develop a more holistic and realistic 
    predictive model for **YouTube content performance**.
    """)

    # Background
    st.subheader("Background")
    st.markdown("""
    The rapid evolution of digital platforms has driven the **cloud computing market** to unprecedented growth, as the demand 
    for scalable data solutions continues to rise.  

    - The **global cloud computing market** is projected to surpass **$2.3 Trillion USD by 2032** (Pangarkar, 2025).  
    - Similarly, **Large Language Models (LLMs)** are also expanding rapidly, expected to reach **$13.52 billion** in market size, 
      with growing adoption in **natural language processing, sentiment analysis, and predictive systems** across industries 
      (Bursuk, 2025).  
    """)

    # Problem Statement
    st.subheader("Problem Statement")
    st.markdown("""
    - Traditional KPIs such as likes, views, and comments may not fully capture the dynamics that influence a video’s success.  
      A study by Abdu et al. (2021) suggests that **advanced sentiment feature extraction** is needed to improve video 
      performance prediction.  

    - YouTube generates **over 500 hours of video per minute**, making **cloud-based solutions essential** for handling 
      data at this scale. However, research on cloud-based solutions remains limited, and further work is needed 
      (Li et al., 2021).  

    - Deploying machine learning models in **cloud environments** presents significant challenges.  
      A study by Sato et al. (2025) identified common issues in **MLOps adoption**, emphasizing the need for integrated 
      deployment strategies to evaluate and deploy ML models in the cloud.  
    """)

    # Objectives
    st.subheader("Objectives")
    st.markdown("""
    1. To investigate which **LLMs** and **predictive models** best predict YouTube video performance.  
    2. To develop a **cloud-based pipeline** using an LLMs-enhanced prediction model.  
    3. To evaluate and deploy the **YouTube video performance predictive model** in the cloud.  
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

    tab1, tab2, tab3 = st.tabs(["Project Overview", "Dashboard", "Sentiment Analysis"])

    # --- TAB 1: Project Overview ---
    with tab1:
        # -- Dataset Overview
        st.subheader("Dataset Overview")
        st.dataframe(df_raw.head(5))
        st.markdown("""
        - Data is extracted from **YouTube API V3** and downloaded from 
          [Kaggle](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset?select=US_youtube_trending_data.csv).  
        - The dataset consists of **268,787 entries with 12 attributes**.  
        - Data covers the period from **2020–2024** in the **US region**.  
        - Target variables are fixed as **view_count** and **likes** to narrow down the study.  
        """)
    
        # -- Processed Dataset (Encodings & LLM Embedded) Overview
        st.subheader("Processed Dataset (LLM Embedded) Overview")
        st.dataframe(filtered_df.head(5))
        st.markdown("""
        - Dataset was processed in **BigQuery** for LLM sentiment scoring on the columns:  
          `title`, `description`, and `tags`.  
        - Encoding techniques were also applied for non-numerical columns.  
        """)
    
        # -- EDA
        st.subheader("Exploratory Data Analysis")
        st.image("EDA.png")
    
        # -- EDA Explanation
        st.markdown("""
        ### EDA Overview
        The analysis is done by first transforming the textual columns into numerical:
    
        - Text data with sentiment values are transformed into sentiment scores using **Google's Gemini LLM** library embedded in BigQuery.  
        - Other text data such as **category** and **days** are transformed using **one-hot encoding techniques**.  
    
        Based on the correlation heatmap:
    
        - **Text sentiments**:  
          - *Tags* and *Title* show high correlation.  
          - *Description* shows low correlation.  
        - **Numerical columns**:  
          - *Comment_count*, *Days_to_trend*, *Dislikes*, and *Views_per_day* show high correlation.  
        - Feature extraction is applied only to features with **high correlation**.  
        """)




     # --- TAB 2: Overview ---
    with tab2:
            # --- Dashboard ---
        st.subheader("Dashboard")
    
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Views", f"{filtered_df['view_count'].max():,.0f}")
        col2.metric("Total Likes", f"{filtered_df['likes'].max():,.0f}")
        col3.metric("Total Comments", f"{filtered_df['comment_count'].max():,.0f}")
    
    
        # --- Engagement Trends Over Time ---
        st.markdown("### Engagement Trends Over Time")
    
        trend_col1, trend_col2 = st.columns(2)
    
        with trend_col1:
            fig_views = px.area(
                filtered_df, x="days_to_trend", y="view_count",
                title="View Count Over Time", markers=True
            )
            fig_views.update_traces(line_color="#1f77b4", fill='tozeroy')
            st.plotly_chart(fig_views, use_container_width=True)
    
        with trend_col2:
            fig_likes = px.bar(
                filtered_df, x="days_to_trend", y="likes",
                title="Likes Distribution Over Time",
                color="likes",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_likes, use_container_width=True)
    
        # Additional Engagement Metrics (Dislikes and Comments)
        trend_col3, trend_col4 = st.columns(2)

        with trend_col3:
            fig_dislikes = px.area(
                filtered_df, x="days_to_trend", y="dislikes",
                title="Dislikes Over Time", markers=True
            )
            fig_dislikes.update_traces(line_color="#d62728", fill='tozeroy')  # Red color
            st.plotly_chart(fig_dislikes, use_container_width=True)
        
        with trend_col4:
            fig_comments = px.bar(
                filtered_df, x="days_to_trend", y="comment_count",
                title="Comments Over Time",
                color="comment_count",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_comments, use_container_width=True)
    
    
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
