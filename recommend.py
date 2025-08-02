import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
def extract_names(x):
    try:
        return ' '.join([i['name'] for i in literal_eval(x)])
    except:
        return ''
def get_director(x):
    try:
        for i in literal_eval(x):
            if i['job'] == 'Director':
                return i['name']
        return ''
    except:
        return ''
movies['tags'] = (
    movies['overview'].fillna('') + ' ' +
    movies['genres'].apply(extract_names) + ' ' +
    movies['keywords'].apply(extract_names) + ' ' +
    movies['cast'].apply(lambda x: ' '.join([i['name'] for i in literal_eval(x)[:3]]) if pd.notnull(x) else '') + ' ' +
    movies['crew'].apply(get_director)
)
movies['tags'] = movies['tags'].str.lower()
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags'])
similarity = cosine_similarity(vectors)
def search_movies(query):
    query = query.lower()
    matches = movies[movies['title'].str.lower().str.contains(query)]
    return matches
def recommend(movie_query):
    matches = search_movies(movie_query)
    if matches.empty:
        print("❌ Movie not found. Try typing a part of the name.")
        return
    print("\n🔍 Found matches:")
    for i, title in enumerate(matches['title'].values[:3]):
        print(f"{i+1}. {title}")
    choice = input("\nEnter the number of the correct movie: ")
    try:
        index = matches.index[int(choice)-1]
    except:
        print("Invalid choice.")
        return
    distances = list(enumerate(similarity[index]))
    recommendations = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    print(f"\n🎬 Movies similar to '{movies.iloc[index].title}':")
    for i in recommendations:
        print(f"• {movies.iloc[i[0]].title}")
query = input("Enter a movie name : ")
recommend(query)
