import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class MovieRecommender:
    def __init__(self, movies_path, credits_path):
        self.movies_df = pd.read_csv(movies_path)
        self.credits_df = pd.read_csv(credits_path)
        self.combined_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        
        self.preprocess_data()
        self.create_similarity_matrix()
    
    def preprocess_data(self):
        """Preprocess and merge datasets"""
        # Basic cleaning
        self.movies_df = self.movies_df.dropna(subset=['overview', 'genres'])
        self.movies_df = self.movies_df.reset_index(drop=True)
        
        def safe_literal_eval(x):
            try:
                return ast.literal_eval(x)
            except:
                return []
        
        # Extract genres and keywords
        self.movies_df['genres_list'] = self.movies_df['genres'].apply(safe_literal_eval)
        self.movies_df['genres_clean'] = self.movies_df['genres_list'].apply(
            lambda x: ' '.join([genre['name'].lower() for genre in x]) if isinstance(x, list) else ''
        )
        
        self.movies_df['keywords_list'] = self.movies_df['keywords'].apply(safe_literal_eval)
        self.movies_df['keywords_clean'] = self.movies_df['keywords_list'].apply(
            lambda x: ' '.join([keyword['name'].lower().replace(' ', '') for keyword in x]) if isinstance(x, list) else ''
        )
        
        # Process credits
        self.credits_df['cast_list'] = self.credits_df['cast'].apply(safe_literal_eval)
        self.credits_df['crew_list'] = self.credits_df['crew'].apply(safe_literal_eval)
        
        def get_top_cast(cast_list):
            if isinstance(cast_list, list):
                return ' '.join([person['name'].lower().replace(' ', '') for person in cast_list[:3]])
            return ''
        
        def get_director(crew_list):
            if isinstance(crew_list, list):
                for person in crew_list:
                    if person['job'] == 'Director':
                        return person['name'].lower().replace(' ', '')
            return ''
        
        self.credits_df['top_cast'] = self.credits_df['cast_list'].apply(get_top_cast)
        self.credits_df['director'] = self.credits_df['crew_list'].apply(get_director)
        
        # Merge datasets
        self.combined_df = self.movies_df.merge(
            self.credits_df[['movie_id', 'top_cast', 'director']], 
            left_on='id', 
            right_on='movie_id', 
            how='left'
        )
        
        # Create features for similarity
        self.combined_df['combined_features'] = (
            self.combined_df['overview'].fillna('') + ' ' +
            self.combined_df['genres_clean'] + ' ' +
            self.combined_df['keywords_clean'] + ' ' +
            self.combined_df['top_cast'].fillna('') + ' ' +
            self.combined_df['director'].fillna('')
        )
    
    def create_similarity_matrix(self):
        """Create similarity matrix"""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.combined_df['combined_features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        self.indices = pd.Series(
            self.combined_df.index, 
            index=self.combined_df['title']
        ).drop_duplicates()
    
    def get_recommendations(self, title, top_n=10):
        """Get recommendations based on movie title"""
        try:
            idx = self.indices[title]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]
            movie_indices = [i[0] for i in sim_scores]
            
            return self.combined_df.iloc[movie_indices][[
                'title', 'genres', 'vote_average', 'release_date', 'overview'
            ]]
            
        except KeyError:
            return f"Movie '{title}' not found. Please check the spelling."
    
    def recommend_by_genre(self, genre, top_n=10):
        """Recommend by genre"""
        genre_lower = genre.lower()
        genre_movies = self.combined_df[
            self.combined_df['genres_clean'].str.contains(genre_lower, na=False)
        ]
        
        if len(genre_movies) == 0:
            return f"No {genre} movies found."
        
        return genre_movies.nlargest(top_n, ['vote_average', 'popularity'])[[
            'title', 'genres', 'vote_average', 'release_date', 'overview'
        ]]
    
    def recommend_by_mood(self, mood, top_n=10):
        """Recommend by mood"""
        mood_genres = {
            'sad': ['Comedy', 'Animation', 'Family'],
            'happy': ['Comedy', 'Animation', 'Adventure'],
            'romantic': ['Romance', 'Drama'],
            'bored': ['Action', 'Adventure', 'Thriller'],
            'stressed': ['Comedy', 'Family', 'Animation'],
            'angry': ['Comedy', 'Animation'],
            'tired': ['Comedy', 'Family'],
            'scared': ['Horror', 'Thriller'],
            'fantasy': ['Fantasy', 'Adventure'],
            'action': ['Action', 'Adventure', 'Thriller']
        }
        
        target_genres = mood_genres.get(mood, ['Comedy'])
        mood_movies = self.combined_df[
            self.combined_df['genres_clean'].str.contains('|'.join([g.lower() for g in target_genres]), na=False)
        ]
        
        if len(mood_movies) == 0:
            return f"No movies found for {mood} mood."
        
        return mood_movies.nlargest(top_n, ['vote_average', 'popularity'])[[
            'title', 'genres', 'vote_average', 'release_date', 'overview'
        ]]
    
    def get_popular_movies(self, top_n=10):
        """Get popular movies"""
        return self.combined_df.nlargest(top_n, 'popularity')[[
            'title', 'genres', 'vote_average', 'release_date', 'overview'
        ]]
    
    def search_movies(self, query, top_n=5):
        """Search movies"""
        query_lower = query.lower()
        
        title_matches = self.combined_df[
            self.combined_df['title'].str.lower().str.contains(query_lower, na=False)
        ]
        
        overview_matches = self.combined_df[
            self.combined_df['overview'].str.lower().str.contains(query_lower, na=False)
        ]
        
        all_matches = pd.concat([title_matches, overview_matches]).drop_duplicates()
        
        if len(all_matches) == 0:
            return f"No movies found for '{query}'."
        
        return all_matches.nlargest(top_n, ['vote_average', 'popularity'])[[
            'title', 'genres', 'vote_average', 'release_date', 'overview'
        ]]