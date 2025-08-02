import pandas as pd
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import pycountry

st.set_page_config(layout="wide")
st.title("🎬 Movie Recommender")

TMDB_API_KEY ="b1b1dc89770344f6675d558c42205f9f" 

@st.cache_resource
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    movies = movies[[
        "movie_id", "title", "overview", "genres", "keywords", "cast", "crew",
        "original_language", "release_date", "vote_average"
    ]]

    def extract_names(x):
        try:
            return " ".join([i["name"] for i in literal_eval(x)])
        except:
            return ""

    def get_director(x):
        try:
            for i in literal_eval(x):
                if i["job"] == "Director":
                    return i["name"]
        except:
            return ""
        return ""

    movies["tags"] = (
        movies["overview"].fillna("")
        + " " + movies["genres"].apply(extract_names)
        + " " + movies["keywords"].apply(extract_names)
        + " " + movies["cast"].apply(lambda x: " ".join([i["name"] for i in literal_eval(x)[:3]]) if pd.notnull(x) else "")
        + " " + movies["crew"].apply(get_director)
    ).str.lower()

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"])
    similarity = cosine_similarity(vectors)

    return movies, similarity


movies, similarity = load_data()

def fetch_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        response = requests.get(url)
        data = response.json()
        if data["results"] and data["results"][0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + data["results"][0]["poster_path"]
    except:
        pass
    return "https://via.placeholder.com/300x450?text=No+Poster"

def get_language_name(code):
    try:
        return pycountry.languages.get(alpha_2=code).name
    except:
        return code

def recommend(selected_title):
    index = movies[movies["title"] == selected_title].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recs = []
    for i in sorted_movies:
        data = movies.iloc[i[0]]

        try:
            genres = ", ".join([g["name"] for g in literal_eval(data["genres"])])
        except:
            genres = "N/A"

        try:
            cast = ", ".join([a["name"] for a in literal_eval(data["cast"])[:3]])
        except:
            cast = "N/A"

        try:
            crew = literal_eval(data["crew"])
            director = next((p["name"] for p in crew if p["job"] == "Director"), "N/A")
        except:
            director = "N/A"

        try:
            year = pd.to_datetime(data["release_date"]).year
        except:
            year = "N/A"

        recs.append({
            "title": data["title"],
            "poster_url": fetch_poster(data["title"]),
            "rating": data.get("vote_average", "N/A"),
            "genres": genres,
            "overview": data.get("overview", "No description."),
            "director": director,
            "cast": cast,
            "year": year,
            "language": get_language_name(data.get("original_language", ""))
        })
    return recs

with st.sidebar:
    st.header("🎯 Filters")

    years = sorted(movies["release_date"].dropna().apply(lambda x: pd.to_datetime(x).year).unique())
    year_filter = st.selectbox("📅 Release Year", options=["All"] + list(map(str, years)))

    language_code_overrides = {
        "cn": "Chinese",
        "xx": None       
    }

    def get_language_name_safe(code):
        if code in language_code_overrides:
            return language_code_overrides[code]
        try:
            return pycountry.languages.get(alpha_2=code).name
        except:
            return None  

    languages = sorted(movies["original_language"].dropna().unique())
    full_languages = []
    for code in languages:
        name = get_language_name_safe(code)
        if name:
            full_languages.append(name)

    language_filter = st.selectbox("🌐 Language", options=["All"] + full_languages)

    st.header("❤️ Favorites")
    if "favorites" not in st.session_state:
        st.session_state.favorites = []

    for i, fav in enumerate(st.session_state.favorites):
        st.image(fav["poster_url"], use_container_width=True)
        st.markdown(f"**🎬 {fav['title']} ({fav['year']})**")
        st.markdown(f"⭐ {fav['rating']} | 🎭 {fav['genres']} | 🌐 {fav['language']}")
        if st.button("❌ Remove", key=f"remove_{fav['title']}"):
            st.session_state.favorites.pop(i)
            st.rerun()

recommendations = []
movie_input = st.text_input("Enter a movie name:")

if movie_input:
    matches = movies[movies["title"].str.lower().str.contains(movie_input.lower())]
    if not matches.empty:
        selected_title = st.selectbox("Select the correct movie:", matches["title"].tolist())
        if selected_title:
            recommendations = recommend(selected_title)

            if year_filter != "All":
                recommendations = [r for r in recommendations if str(r["year"]) == year_filter]
            if language_filter != "All":
                recommendations = [r for r in recommendations if r["language"] == language_filter]

if recommendations:
    st.subheader(f"🎬 Recommendations for '{selected_title}'")
    cols = st.columns(3)
    for idx, movie in enumerate(recommendations):
        col = cols[idx % 3]
        with col:
            st.image(movie["poster_url"], use_container_width=True)
            st.markdown(f"**{movie['title']} ({movie['year']})**")
            st.markdown(f"⭐ {movie['rating']} | 🎭 {movie['genres']}")
            st.markdown(f"🎥 *{movie['director']}*")
            st.markdown(f"👥 *{movie['cast']}*")
            st.markdown(f"📝 {movie['overview'][:100]}...")

            if movie not in st.session_state.favorites:
                if st.button("💾 Add to Favorites", key=f"fav_{movie['title']}"):
                    st.session_state.favorites.append(movie)
                    st.rerun()
else:
    st.info("No recommendations yet. Start by typing a movie name above.")
