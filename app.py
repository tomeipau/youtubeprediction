def show_analysis():
    st.title("Analysis & Video Performance Dashboard")

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

    # Tab order switched
    tab1, tab2, tab3 = st.tabs(["Project Overview", "Video Overview", "Sentiment Analysis"])

    # --- TAB 1: Now Project Overview ---
    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(filtered_df.head(5))
        st.markdown("""
        Data is extracted from YouTube API V3...
        """)

        st.subheader("Processed Dataset (LLM Embedded) Overview")
        st.dataframe(filtered_df.head(5))
        st.markdown("""
        Dataset was processed in BigQuery for LLM sentiment scoring...
        """)

        st.subheader("Exploratory Data Analysis")
        st.image("EDA.png")
        st.markdown("""
        This app predicts future views and likes...
        """)

    # --- TAB 2: Now Video Overview ---
    with tab2:
        st.subheader("Performance Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Views", f"{filtered_df['view_count'].max():,.0f}")
        col2.metric("Total Likes", f"{filtered_df['likes'].max():,.0f}")
        col3.metric("Total Comments", f"{filtered_df['comment_count'].max():,.0f}")

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
