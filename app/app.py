import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. LOAD AND PREPARE DATA ---
# Load the dataset
try:
    # Make sure the CSV file is in the same folder as your app.py file
    df = pd.read_csv('IMDB-Movie-Dataset(2023-1951).csv')
except FileNotFoundError:
    st.error("Dataset file not found. Please make sure 'IMDB-Movie-Dataset(2023-1951).csv' is in the same folder as your app.py file.")
    st.stop() # Stop the app from running further

# Data Preprocessing and Feature Engineering
df['genre'] = df['genre'].fillna('')
df['overview'] = df['overview'].fillna('')
df['director'] = df['director'].fillna('')
df['cast'] = df['cast'].fillna('')
df['tags'] = df['genre'] + ' ' + df['overview'] + ' ' + df['director'] + ' ' + df['cast']

# Text Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
movie_vectors = tfidf.fit_transform(df['tags']).toarray()

# Calculate Cosine Similarity
similarity_matrix = cosine_similarity(movie_vectors)

# --- 2. MODIFIED RECOMMENDATION FUNCTION ---
def recommend(movie_title):
    """
    Recommends movies similar to the input movie_title.
    Returns a list of recommended movie names.
    """
    try:
        movie_index = df[df['movie_name'] == movie_title].index[0]
        distances = similarity_matrix[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

        recommended_movies = []
        for i in movies_list[1:6]:
            recommended_movies.append(df.iloc[i[0]].movie_name)
        return recommended_movies
            
    except IndexError:
        return ["Movie not found in the dataset."]

# --- 3. STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("🎬 Bollywood Movie Recommendation System")

movie_list = df['movie_name'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown to get a recommendation",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movies = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    
    # Create columns for a more organized layout
    cols = st.columns(5)
    for idx, movie in enumerate(recommended_movies):
        with cols[idx]:
            st.text(movie)