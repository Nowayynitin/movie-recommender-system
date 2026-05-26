import pickle
import streamlit as st
import requests

st.set_page_config(
    page_title="Movie Recommender System",
    layout="wide"
)

@st.cache_data
def fetch_poster(movie_id):

    try:

        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"

        response = requests.get(url, timeout=10)

        data = response.json()

        poster_path = data.get('poster_path')

        if poster_path:

            return "https://image.tmdb.org/t/p/w500/" + poster_path

        else:

            return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"

    except:

        return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"


movies = pickle.load(open('movies.pkl', 'rb'))

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)


def recommend(movie, n=10):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        enumerate(distances),
        reverse=True,
        key=lambda x: x[1]
    )[1:n+1]

    recommended_movie_names = []

    recommended_movie_posters = []

    for i in movies_list:

        movie_id = movies.iloc[i[0]].movie_id

        recommended_movie_names.append(
            movies.iloc[i[0]].title
        )

        recommended_movie_posters.append(
            fetch_poster(movie_id)
        )

    return recommended_movie_names, recommended_movie_posters


st.title("🎬 Movie Recommender System")

st.write("Get top movie recommendations instantly")


movie_list = movies['title'].values

selected_movie = st.selectbox(
    "Type or select a movie from dropdown",
    movie_list
)

number_of_movies = st.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)


if st.button('Show Recommendation'):

    with st.spinner('Fetching recommendations...'):

        recommended_movie_names, recommended_movie_posters = recommend(
            selected_movie,
            number_of_movies
        )

        cols = st.columns(5)

        for idx in range(number_of_movies):

            with cols[idx % 5]:

                st.text(recommended_movie_names[idx])

                st.image(
                    recommended_movie_posters[idx],
                    use_container_width=True
                )