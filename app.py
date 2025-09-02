import streamlit as st
import pandas as pd
import joblib
import base64

# --------------------------
# Load Data & Models
# --------------------------
df = pd.read_csv("youtube_data_deploy.csv")
model_likes = joblib.load("boosted_tree_model_likes.pkl")
model_viewcount = joblib.load("boosted_tree_model_viewcount.pkl")

# --------------------------
# Streamlit Config
# --------------------------
st.set_page_config(page_title="YouTube Performance Predictor", layout="wide")

# --------------------------
# Tabs Layout
# --------------------------
tab1, tab2 = st.tabs(["📄 Project Overview", "📊 Video Overview"])

# --------------------------
# TAB 1 - PROJECT OVERVIEW
# --------------------------
with tab1:
    st.subheader("Project Overview")

    pdf_file_path = "streamlit_intro.pdf"  # <-- Must be placed in repository
    try:
        with open(pdf_file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("PDF file not found. Please upload 'streamlit_intro.pdf' to the repository.")

# --------------------------
# TAB 2 - VIDEO OVERVIEW
# --------------------------
with tab2:
    st.header("Predict YouTube Video Performance")
    video_url = st.text_input("Enter YouTube Video URL")

    if video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]
        video_row = df[df['video_id'] == video_id]

        if video_row.empty:
            st.error("Video not found in dataset.")
        else:
            # Extract values
            view_count = int(video_row['view_count'].values[0])
            likes = int(video_row['likes'].values[0])

            # Prepare features for prediction
            features = pd.DataFrame([[view_count, likes]], columns=["view_count", "likes"])
            predicted_views = model_viewcount.predict(features)[0]
            predicted_likes = model_likes.predict(features)[0]

            # Display predictions
            st.success("Predicted Performance")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Views", f"{predicted_views:,.0f}", f"Current: {view_count:,.0f}")
            col2.metric("Predicted Likes", f"{predicted_likes:,.0f}", f"Current: {likes:,.0f}")
