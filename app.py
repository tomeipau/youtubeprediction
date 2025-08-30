import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("youtube_data.csv")

# Sidebar - User Input
st.sidebar.header("Filter Options")
channel_selection = st.sidebar.multiselect(
    "Select Channel(s)", options=df["channelTitle"].unique()
)
video_link_input = st.sidebar.text_input("Paste YouTube Video Link (Optional)")

# Filter Logic
filtered_df = df.copy()
if channel_selection:
    filtered_df = filtered_df[filtered_df["channelTitle"].isin(channel_selection)]
if video_link_input:
    filtered_df = filtered_df[filtered_df["youtube_link"].str.contains(video_link_input)]

st.title("YouTube Performance Analysis Dashboard")

# ---------------------- KPI Cards ----------------------
avg_views = int(filtered_df['view_count'].mean()) if not filtered_df.empty else 0
avg_likes_per_view = round(filtered_df['likes_per_view'].mean(), 4) if not filtered_df.empty else 0
best_sentiment = (
    filtered_df[['title_score','description_score','tags_score']]
    .mean().idxmax() if not filtered_df.empty else "N/A"
)

col1, col2, col3 = st.columns(3)
col1.metric("Average Views", avg_views)
col2.metric("Avg Likes per View", avg_likes_per_view)
col3.metric("Strongest Sentiment Driver", best_sentiment)

st.markdown("---")

# ---------------------- Sentiment vs Engagement Impact ----------------------
if not filtered_df.empty:
    fig_sentiment_impact = px.scatter(
        filtered_df,
        x="title_score",
        y="view_count",
        size="likes_per_view",
        color="description_score",
        hover_data=["tags_score", "youtube_link"],
        title="Impact of Sentiment Scores on View Count"
    )
    st.plotly_chart(fig_sentiment_impact, use_container_width=True)

# ---------------------- Top Performing Videos ----------------------
st.subheader("Top Performing Videos")
if not filtered_df.empty:
    top_videos = (
        filtered_df[['youtube_link', 'view_count', 'likes', 'likes_per_view']]
        .sort_values(by='view_count', ascending=False)
        .head(10)
    )
    st.dataframe(top_videos)

# ---------------------- Sentiment Distribution with Engagement Bands ----------------------
if not filtered_df.empty:
    filtered_df["engagement_band"] = pd.qcut(filtered_df["likes_per_view"], q=3, labels=["Low", "Medium", "High"])
    fig_sentiment_band = px.histogram(
        filtered_df, x="title_score", color="engagement_band",
        nbins=30, barmode="overlay",
        title="Sentiment vs Engagement Levels"
    )
    st.plotly_chart(fig_sentiment_band, use_container_width=True)

# ---------------------- Day-of-Week or Month Trends ----------------------
if not filtered_df.empty:
    # If month_encoded & day_encoded exist
    if "month_encoded" in filtered_df.columns and "day_encoded" in filtered_df.columns:
        st.subheader("Upload Performance by Day & Month")
        fig_day = px.bar(
            filtered_df.groupby("day_encoded")["view_count"].mean().reset_index(),
            x="day_encoded", y="view_count",
            title="Average Views by Day of the Week"
        )
        st.plotly_chart(fig_day, use_container_width=True)

        fig_month = px.bar(
            filtered_df.groupby("month_encoded")["view_count"].mean().reset_index(),
            x="month_encoded", y="view_count",
            title="Average Views by Month"
        )
        st.plotly_chart(fig_month, use_container_width=True)
