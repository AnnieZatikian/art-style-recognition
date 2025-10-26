# Import libraries
import streamlit as st
import pandas as pd
import requests
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
import random

# Load CSVs
artists_df = pd.read_csv('data/artists.csv')
art_styles_df = pd.read_csv('data/art_style.csv')

# Streamlit Page Settings
st.set_page_config(
    page_title="Art Style Explorer 🎨",
    page_icon="🎨",
    layout="wide"
)

# Unified dark forest green styling
st.markdown("""
    <style>
    .stApp {
        background-color: #032d21 !important;
        color: #ffffff;
    }

    [data-testid="stSidebar"] {
        background-color: #032d21 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stFileUploader {
        background-color: #04382a !important;
        color: #ffffff !important;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🎨 ArtStyle Explorer")
selection = st.sidebar.radio("Choose a section:", ["Artists", "Art Styles"])

# Artists Section
if selection == "Artists":
    st.title("🎨 Discover Artists")
    st.markdown("**Select an artist from the sidebar to explore their story.**")

    artist_names = artists_df['name'].tolist()
    selected_artist = st.sidebar.selectbox("Select an artist", ["Select an artist"] + artist_names)

    if selected_artist == "Select an artist":
        st.markdown("---")
        st.image("data/front_image.png", use_column_width=True)
        st.markdown("""
            <h2 style="text-align: center; color: #a6f4c5;">Welcome to the ArtStyle Explorer</h2>
            <p style="text-align: center; color: #ccc;">Discover amazing artists, learn about their lives, and explore different art styles from across history.</p>
            """, unsafe_allow_html=True)
    else:
        artist_info = artists_df[artists_df['name'] == selected_artist].iloc[0]
        st.markdown(f"<h2 style='text-align: center; color: #a6f4c5;'>{selected_artist}</h2>", unsafe_allow_html=True)
        st.write("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            wiki_link = artist_info['wikipedia']
            if pd.notna(wiki_link) and "wikipedia.org" in wiki_link:
                try:
                    title = wiki_link.split("/wiki/")[-1]
                    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
                    response = requests.get(api_url)
                    if response.status_code == 200:
                        data = response.json()
                        if 'thumbnail' in data:
                            img_url = data['thumbnail']['source']
                            st.image(img_url, caption=selected_artist)
                except Exception:
                    st.write("Image could not be loaded.")

        with col2:
            if 'bio' in artist_info and pd.notna(artist_info['bio']):
                st.markdown(f"<p style='color:#ccc'>{artist_info['bio']}</p>", unsafe_allow_html=True)
            else:
                st.write("Biography not available.")
            if pd.notna(wiki_link):
                st.markdown(f"[Read more on Wikipedia]({wiki_link})", unsafe_allow_html=True)

# Art Styles Section
elif selection == "Art Styles":
    st.title("🎯 Art Style Prediction & Exploration")

    st.markdown("## 🔮 Style Predictor")
    with st.container():
        st.markdown("""
        <div style="padding:1.5em; background-color:#04382a; border-radius:10px;">
            <h4 style="color:#ffffff;">Upload a painting to predict its art style using your trained model 🎨</h4>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload a painting (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption=" Uploaded Image", width=300)


            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224)) / 255.0
            img = np.expand_dims(img, axis=0)

            try:
                model = tf.keras.models.load_model('models/art_style_classifier.h5')
                with open('models/label_encoder.pkl', 'rb') as f:
                    label_encoder = pickle.load(f)

                prediction = model.predict(img)[0]
                top_indices = np.argsort(prediction)[-3:][::-1]
                top_labels = label_encoder.inverse_transform(top_indices)
                top_scores = prediction[top_indices]

                # Main prediction
                predicted_class = top_labels[0]
                confidence = top_scores[0]

                st.markdown(f"""
                <div style="padding:1em; background-color:#09341f; border-radius:10px;">
                    <h3 style="color:#a6f4c5;">🎯 Predicted Style: <u>{predicted_class}</u></h3>
                    <p style="color:#ccc;">Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

                #  Show description
                desc = art_styles_df[art_styles_df['style'] == predicted_class]['description'].values
                if len(desc) > 0:
                    st.markdown(f"<p style='color:#ccc; margin-top:1em;'>{desc[0]}</p>", unsafe_allow_html=True)

                # 🖼 Show example image
                sample_dir = f"data/images/{predicted_class.replace(' ', '_')}"
                if os.path.isdir(sample_dir):
                    sample_imgs = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if sample_imgs:
                        sample_img_path = os.path.join(sample_dir, random.choice(sample_imgs))
                        st.markdown("### 🖼 Example of This Style")
                        st.image(sample_img_path, caption="🎨 Example Artwork", width=300)

                #  Top 3 predictions
                st.markdown("### 📊 Top 3 Style Predictions")
                st.bar_chart(pd.DataFrame({'Confidence': top_scores}, index=top_labels))

            except Exception as e:
                st.error("⚠️ Could not load model or make prediction.")
                st.text(str(e))

    # 🔍 Search section (optional)
    st.markdown("---")
    st.subheader("🔎 Search Art Styles")
    search_column = st.selectbox("Choose a column to search in:", art_styles_df.columns)
    search_term = st.text_input(f"Enter a search term for {search_column}")

    if search_term:
        filtered_df = art_styles_df[art_styles_df[search_column].astype(str).str.contains(search_term, case=False, na=False)]
        st.subheader("🔍 Search Results")
        if not filtered_df.empty:
            st.dataframe(filtered_df)
        else:
            st.write("No results found. Try another search term.")
