import re
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

class CompleteMovieExpert:
    def __init__(self, recommender):
        self.recommender = recommender
        self.setup_intent_patterns()
        self.setup_responses()
        self.conversation_history = []
    
    def setup_intent_patterns(self):
        """Define patterns for ALL possible user intents"""
        self.intent_patterns = {
            #  GENRE-BASED REQUESTS
            'action_movies': [
                r'action|action movies|action films|action genre',
                r'recommend action|suggest action|give action',
                r'action.*movie|movie.*action',
                r'explosion|fight|adventure|thriller|combat',
                r'fast.*furious|mission impossible|john wick|expendables',
                r'fighting|battle|war.*movie',
                r'want action|need action|looking for action'
            ],
            'romantic_movies': [
                r'romantic|romance|romantic movies|romance films',
                r'love story|love films|rom com|romantic comedy',
                r'relationship|date movie|couple movie',
                r'heart.*warming|emotional.*love',
                r'valentine|anniversary|date night',
                r'chick flick|love triangle'
            ],
            'comedy_movies': [
                r'comedy|comedy movies|funny films|humor',
                r'make me laugh|funny movie|hilarious',
                r'comedy.*movie|movie.*comedy',
                r'joke|humorous|light.*hearted',
                r'stand.*up|sitcom|comedy show'
            ],
            'horror_movies': [
                r'horror|horror movies|scary films|frightening',
                r'ghost|zombie|vampire|monster|haunted',
                r'thriller|suspense|psychological',
                r'paranormal|supernatural|dark',
                r'nightmare|terror|fear'
            ],
            'sci-fi_movies': [
                r'sci-fi|science fiction|sci fi',
                r'future|space|alien|robot|android',
                r'star wars|star trek|avatar|matrix',
                r'technology|futuristic|time travel',
                r'galaxy|universe|cosmic'
            ],
            'drama_movies': [
                r'drama|dramatic|emotional',
                r'serious.*movie|intense.*film',
                r'story.*driven|character.*driven',
                r'real life|slice of life'
            ],
            'fantasy_movies': [
                r'fantasy|magical|mythical',
                r'wizard|witch|dragon|magic',
                r'fairy tale|mythology|legend',
                r'harry potter|lord of the rings'
            ],
            'animation_movies': [
                r'animation|animated|cartoon',
                r'pixar|disney|anime',
                r'kids movie|family animation',
                r'animated.*film'
            ],
            'family_movies': [
                r'family|family movies|kids films',
                r'children|child friendly|pg rated',
                r'watch with kids|family night'
            ],
            'documentary_movies': [
                r'documentary|docu|real story',
                r'non.*fiction|true story|biography',
                r'educational|informative'
            ],
            
            #  MOOD-BASED REQUESTS
            'sad_mood': [
                r'sad|depressed|down|unhappy|blue',
                r'feeling low|not good|bad mood',
                r'heartbroken|broken.*heart',
                r'miserable|gloomy|upset'
            ],
            'happy_mood': [
                r'happy|joyful|cheerful|excited',
                r'good mood|feeling great|amazing',
                r'celebrat|party|festive'
            ],
            'bored_mood': [
                r'bored|boring|nothing to do',
                r'uninterested|dull|monotonous'
            ],
            'stressed_mood': [
                r'stressed|anxious|worried|pressure',
                r'tense|nervous|overwhelmed'
            ],
            'romantic_mood': [
                r'romantic mood|feeling romantic',
                r'in love|loving|affectionate'
            ],
            'energetic_mood': [
                r'energetic|energized|pumped',
                r'active|hyper|full of energy'
            ],
            'relaxed_mood': [
                r'relaxed|calm|peaceful|chill',
                r'laid back|easy.*going|serene'
            ],
            
            #  SPECIAL OCCASIONS
            'birthday': [
                r'birthday|bday|born today',
                r'my birthday|birthday.*today',
                r'celebrat|party|anniversary'
            ],
            'date_night': [
                r'date night|date.*movie',
                r'with partner|with girlfriend|with boyfriend',
                r'couple.*movie|romantic.*evening'
            ],
            'friends_hangout': [
                r'with friends|friends.*hangout',
                r'group.*movie|party.*movie'
            ],
            'family_time': [
                r'with family|family.*time',
                r'parents|children|kids'
            ],
            'alone_time': [
                r'alone|by myself|solo',
                r'me time|personal.*time'
            ],
            
            #  SEARCH-BASED
            'similar_movies': [
                r'movies like|similar to|like.*movie',
                r'recommend.*like|suggest.*like',
                r'comparable to|same as'
            ],
            'actor_movies': [
                r'movies with|films with',
                r'actor.*movie|starring',
                r'featuring.*actor'
            ],
            'director_movies': [
                r'director.*movie|movie.*director',
                r'directed by|films by'
            ],
            'year_movies': [
                r'movies from|films from',
                r'released in|year.*movie',
                r'2023|2022|2021|2020'
            ],
            'popular_movies': [
                r'popular movies|trending films',
                r'best.*movies|top.*films',
                r'hit movies|blockbuster'
            ],
            'award_movies': [
                r'oscar|award.*winning',
                r'academy award|best picture'
            ],
            
            #  GENERAL CONVERSATION
            'greeting': [
                r'hi|hello|hey|hola|greetings',
                r'how are you|what\'s up|howdy'
            ],
            'thanks': [
                r'thank|thanks|appreciate|grateful'
            ],
            'farewell': [
                r'bye|goodbye|see you|farewell'
            ],
            'help': [
                r'help|what can you do|how to use',
                r'commands|options|features'
            ],
            'joke_request': [
                r'joke|funny|make me laugh',
                r'humor|entertain me'
            ],
            'fact_request': [
                r'fact|interesting|tell me about',
                r'teach me|share knowledge'
            ],
            'story_request': [
                r'story|tell me a story',
                r'narrative|tale'
            ],
            'time_request': [
                r'time|what time|current time',
                r'clock|what.*o.*clock'
            ],
            'date_request': [
                r'date|what date|today.*date',
                r'day|what day'
            ],
            
            #  MULTILINGUAL (BENGALI)
            'bengali_movies': [
                r'সিনেমা|মুভি|চলচ্চিত্র',
                r'দেখব|দেখতে|দেখা',
                r'রেকমেন্ড|সাজেস্ট|বলো'
            ],
            'bengali_sad': [
                r'মন খারাপ|দু:খিত|খারাপ লাগছে',
                r'কাবাব|মন ভারী|মন ভালো না'
            ],
            'bengali_happy': [
                r'খুশি|আনন্দ|ভালো লাগছে',
                r'মন ভালো|হাসিখুশি'
            ]
        }
    
    def setup_responses(self):
        """Setup response templates for ALL scenarios"""
        self.response_templates = {
            # Genre responses
            'action_movies': " **Action Movie Recommendations!** \n\nGet ready for adrenaline-pumping action! Here are the best action films:\n\n",
            'romantic_movies': " **Romantic Movie Recommendations!** \n\nPerfect for love stories! Here are beautiful romance films:\n\n",
            'comedy_movies': " **Comedy Movie Recommendations!** \n\nTime for laughter! Here are hilarious comedy movies:\n\n",
            'horror_movies': " **Horror Movie Recommendations!** \n\nFeeling brave? Here are scary horror films:\n\n",
            'sci-fi_movies': " **Sci-Fi Movie Recommendations!** \n\nReady for adventure! Here are amazing sci-fi movies:\n\n",
            'drama_movies': " **Drama Movie Recommendations!** \n\nEmotional and powerful stories! Here are great drama films:\n\n",
            'fantasy_movies': " **Fantasy Movie Recommendations!** \n\nMagical worlds await! Here are enchanting fantasy films:\n\n",
            'animation_movies': " **Animation Movie Recommendations!** \n\nAnimated wonders! Here are fantastic animated films:\n\n",
            'family_movies': " **Family Movie Recommendations!** \n\nPerfect for everyone! Here are family-friendly films:\n\n",
            'documentary_movies': " **Documentary Recommendations!** \n\nReal stories! Here are informative documentaries:\n\n",
            
            # Mood responses
            'sad_mood': " I understand you're feeling down. Let me cheer you up with some uplifting movies!\n\n",
            'happy_mood': " Great to hear you're happy! Let's keep the good vibes going with these films!\n\n",
            'bored_mood': " Boredom is no fun! Let me suggest some exciting movies to spark your interest!\n\n",
            'stressed_mood': " I know stress can be tough. Here are some relaxing movies to help you unwind!\n\n",
            'romantic_mood': " Romance is in the air! Here are beautiful love stories for you!\n\n",
            'energetic_mood': " Full of energy! Let's channel that into some action-packed movies!\n\n",
            'relaxed_mood': " Perfect relaxed mood! Here are some calming movies to match!\n\n",
            
            # Special occasions
            'birthday': " **Happy Birthday!** \n\nSpecial day calls for special movies! Here are celebratory films:\n\n",
            'date_night': " **Perfect Date Night Movies!** \n\nRomantic films for a wonderful evening:\n\n",
            'friends_hangout': " **Great Movies with Friends!** \n\nFun films perfect for group watching:\n\n",
            'family_time': " **Family Movie Night!** \n\nMovies everyone will enjoy together:\n\n",
            'alone_time': " **Perfect Solo Movies!** \n\nGreat films for some quality me-time:\n\n"
        }
    
    def detect_intent(self, user_input):
        """Detect user intent from ANY input"""
        user_input_lower = user_input.lower().strip()
        
        # First check for exact matches
        exact_matches = {
            'action': 'action_movies',
            'action movies': 'action_movies',
            'romantic': 'romantic_movies',
            'romance': 'romantic_movies',
            'comedy': 'comedy_movies',
            'horror': 'horror_movies',
            'sci-fi': 'sci-fi_movies',
            'drama': 'drama_movies',
            'fantasy': 'fantasy_movies',
            'animation': 'animation_movies',
            'family': 'family_movies',
            'documentary': 'documentary_movies',
            'sad': 'sad_mood',
            'happy': 'happy_mood',
            'bored': 'bored_mood',
            'stressed': 'stressed_mood',
            'birthday': 'birthday',
            'date': 'date_night'
        }
        
        if user_input_lower in exact_matches:
            return exact_matches[user_input_lower], None
        
        # Then check pattern matches
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return intent, re.search(pattern, user_input_lower)
        
        return 'general_conversation', None
    
    def extract_movie_title(self, user_input):
        """Extract movie title for similar recommendations"""
        patterns = [
            r'movies like (.+)',
            r'similar to (.+)',
            r'recommend.*like (.+)',
            r'like (.+)',
            r'comparable to (.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return match.group(1).strip()
        return None
    
    def extract_actor_director(self, user_input):
        """Extract actor or director name"""
        patterns = [
            r'movies with (.+)',
            r'films with (.+)',
            r'actor.*?([a-zA-Z\s]+)',
            r'director.*?([a-zA-Z\s]+)',
            r'starring.*?([a-zA-Z\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return match.group(1).strip()
        return None
    
    def extract_year(self, user_input):
        """Extract year from user input"""
        year_match = re.search(r'(19|20)\d{2}', user_input)
        if year_match:
            return year_match.group()
        return None
    
    def process_query(self, user_input):
        """Main NLP processing - handles ALL question types"""
        user_input_lower = user_input.lower().strip()
        
        # Store conversation history
        self.conversation_history.append(f"User: {user_input}")
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        intent, match = self.detect_intent(user_input)
        
        print(f"DEBUG: User: '{user_input}' | Intent: {intent}")  # For debugging
        
        #  Handle ALL movie genre requests
        if any(intent == genre for genre in ['action_movies', 'romantic_movies', 'comedy_movies', 
                                           'horror_movies', 'sci-fi_movies', 'drama_movies',
                                           'fantasy_movies', 'animation_movies', 'family_movies',
                                           'documentary_movies']):
            return self.handle_genre_request(intent)
        
        #  Handle ALL mood-based requests
        elif any(intent == mood for mood in ['sad_mood', 'happy_mood', 'bored_mood', 'stressed_mood',
                                           'romantic_mood', 'energetic_mood', 'relaxed_mood']):
            return self.handle_mood_request(intent)
        
        #  Handle ALL special occasions
        elif any(intent == occasion for occasion in ['birthday', 'date_night', 'friends_hangout',
                                                   'family_time', 'alone_time']):
            return self.handle_occasion_request(intent)
        
        #  Handle search-based requests
        elif intent == 'similar_movies':
            movie_title = self.extract_movie_title(user_input)
            if movie_title:
                return self.get_similar_movies(movie_title)
            else:
                return "Which movie would you like similar recommendations for? Try: 'movies like Inception'"
        
        elif intent == 'actor_movies' or intent == 'director_movies':
            name = self.extract_actor_director(user_input)
            if name:
                return self.get_movies_by_person(name, 'actor' if intent == 'actor_movies' else 'director')
            else:
                return "Which actor or director are you interested in? Try: 'movies with Tom Cruise'"
        
        elif intent == 'year_movies':
            year = self.extract_year(user_input)
            if year:
                return self.get_movies_by_year(year)
            else:
                return "Which year are you interested in? Try: 'movies from 2020'"
        
        elif intent == 'popular_movies':
            return self.get_popular_movies()
        
        elif intent == 'award_movies':
            return self.get_award_winning_movies()
    
        #  Handle general conversation
        elif intent == 'greeting':
            return random.choice([
                "Hello! I'm your complete movie expert!  I can recommend movies for ANY situation!",
                "Hi there! Tell me what you're looking for - any genre, mood, or occasion!",
                "Hey! I'm here to help you find perfect movies for ANY scenario! What do you need?"
            ])
        
        elif intent == 'thanks':
            return "You're very welcome!  Let me know if you need more recommendations!"
        
        elif intent == 'farewell':
            return "Goodbye!  Hope you enjoy your movie time!"
        
        elif intent == 'help':
            return self.get_help_response()
        
        elif intent == 'joke_request':
            return self.get_joke()
        
        elif intent == 'fact_request':
            return self.get_fact()
        
        elif intent == 'story_request':
            return self.get_story()
        
        elif intent == 'time_request':
            return f" Current time: {datetime.now().strftime('%I:%M %p')}"
        
        elif intent == 'date_request':
            return f" Today is: {datetime.now().strftime('%A, %B %d, %Y')}"
        
        #  Handle Bengali requests
        elif intent in ['bengali_movies', 'bengali_sad', 'bengali_happy']:
            return self.handle_bengali_request(intent, user_input)
        
        # Default: Handle any other query
        else:
            return self.handle_unknown_query(user_input)
    
    def handle_genre_request(self, genre_intent):
        """Handle any genre request"""
        genre_map = {
            'action_movies': 'action',
            'romantic_movies': 'romance',
            'comedy_movies': 'comedy',
            'horror_movies': 'horror',
            'sci-fi_movies': 'science fiction',
            'drama_movies': 'drama',
            'fantasy_movies': 'fantasy',
            'animation_movies': 'animation',
            'family_movies': 'family',
            'documentary_movies': 'documentary'
        }
        
        genre = genre_map.get(genre_intent, 'action')
        movies = self.recommender.recommend_by_genre(genre)
        
        if isinstance(movies, pd.DataFrame) and len(movies) > 0:
            response = self.response_templates.get(genre_intent, f"**{genre.title()} Movie Recommendations!** \n\n")
            return response + self.format_movie_list(movies)
        else:
            return f"I couldn't find any {genre} movies. Try another genre!"
    
    def handle_mood_request(self, mood_intent):
        """Handle any mood request"""
        mood_map = {
            'sad_mood': 'happy',
            'happy_mood': 'happy',
            'bored_mood': 'action',
            'stressed_mood': 'comedy',
            'romantic_mood': 'romance',
            'energetic_mood': 'action',
            'relaxed_mood': 'drama'
        }
        
        target_mood = mood_map.get(mood_intent, 'comedy')
        movies = self.recommender.recommend_by_mood(target_mood)
        
        if isinstance(movies, pd.DataFrame) and len(movies) > 0:
            response = self.response_templates.get(mood_intent, "**Perfect Movies for Your Mood!** \n\n")
            return response + self.format_movie_list(movies)
        else:
            return "Let me recommend some popular movies for you!\n\n" + self.get_popular_movies()
    
    def handle_occasion_request(self, occasion_intent):
        """Handle any special occasion request"""
        occasion_map = {
            'birthday': ['comedy', 'animation', 'musical'],
            'date_night': ['romance', 'comedy', 'drama'],
            'friends_hangout': ['comedy', 'action', 'horror'],
            'family_time': ['family', 'animation', 'comedy'],
            'alone_time': ['drama', 'documentary', 'fantasy']
        }
        
        target_genres = occasion_map.get(occasion_intent, ['comedy'])
        movies = self.recommender.combined_df[
            self.recommender.combined_df['genres_clean'].str.contains('|'.join(target_genres), na=False)
        ]
        
        if len(movies) > 0:
            top_movies = movies.nlargest(8, ['vote_average', 'popularity'])
            response = self.response_templates.get(occasion_intent, f"**Perfect Movies for Your Occasion!** \n\n")
            return response + self.format_movie_list(top_movies)
        else:
            return self.get_popular_movies()
    
    def get_similar_movies(self, movie_title):
        """Get movies similar to given title"""
        try:
            similar_movies = self.recommender.get_recommendations(movie_title)
            if isinstance(similar_movies, pd.DataFrame):
                return f"**Movies similar to '{movie_title}'** \n\n{self.format_movie_list(similar_movies)}"
            else:
                return f"**Movies similar to '{movie_title}'** \n\n{similar_movies}"
        except:
            return f"Sorry, I couldn't find movies similar to '{movie_title}'. Try another movie title!"
    
    def get_movies_by_person(self, name, person_type):
        """Get movies by actor or director"""
        if person_type == 'actor':
            movies = self.recommender.combined_df[
                self.recommender.combined_df['top_cast'].str.contains(name.lower().replace(' ', ''), na=False)
            ]
        else:  # director
            movies = self.recommender.combined_df[
                self.recommender.combined_df['director'].str.contains(name.lower().replace(' ', ''), na=False)
            ]
        
        if len(movies) > 0:
            top_movies = movies.nlargest(8, ['vote_average', 'popularity'])
            return f"**{person_type.title()} {name}'s Movies** \n\n{self.format_movie_list(top_movies)}"
        else:
            return f"Sorry, I couldn't find any movies with {person_type} {name}."
    
    def get_movies_by_year(self, year):
        """Get movies from specific year"""
        movies = self.recommender.combined_df[
            self.recommender.combined_df['release_date'].str.contains(year, na=False)
        ]
        
        if len(movies) > 0:
            top_movies = movies.nlargest(8, ['vote_average', 'popularity'])
            return f"**Movies from {year}** \n\n{self.format_movie_list(top_movies)}"
        else:
            return f"Sorry, I couldn't find any movies from {year}."
    
    def get_popular_movies(self):
        """Get popular movies"""
        popular = self.recommender.get_popular_movies(10)
        return "**Most Popular Movies Right Now!** \n\n" + self.format_movie_list(popular)
    
    def get_award_winning_movies(self):
        """Get award-winning movies"""
        award_movies = self.recommender.combined_df.nlargest(8, 'vote_average')
        return "**Award-Winning & Highly Rated Movies!** \n\n" + self.format_movie_list(award_movies)
    
    def handle_bengali_request(self, intent, user_input):
        """Handle Bengali language requests"""
        if 'মন খারাপ' in user_input or intent == 'bengali_sad':
            movies = self.recommender.recommend_by_mood('happy')
            return "আপনার মন ভালো করার জন্য সিনেমা \n\n" + self.format_movie_list(movies)
        elif 'সিনেমা' in user_input or intent == 'bengali_movies':
            popular = self.recommender.get_popular_movies(8)
            return "সেরা সিনেমা রেকমেন্ডেশন \n\n" + self.format_movie_list(popular)
        else:
            return "আমি আপনার সিনেমা বিশেষজ্ঞ! আপনি কি ধরনের সিনেমা দেখতে চান?"
    
    def handle_unknown_query(self, user_input):
        """Handle any unknown query intelligently"""
        # Check if it's movie-related
        movie_keywords = ['movie', 'film', 'watch', 'see', 'cinema', 'theater']
        if any(keyword in user_input.lower() for keyword in movie_keywords):
            return self.get_popular_movies()
        
        # Default helpful response
        return self.get_help_response()
    
    def format_movie_list(self, movies):
        """Format movie list for display"""
        if isinstance(movies, str):
            return movies
        
        if len(movies) == 0:
            return "No movies found matching your criteria."
        
        response = ""
        for i, (_, movie) in enumerate(movies.iterrows(), 1):
            response += f"{i}. **{movie['title']}** ⭐ {movie['vote_average']}/10\n"
            response += f"   🎭 {self.extract_genres(movie['genres'])}\n"
            response += f"   📅 {movie['release_date']}\n"
            response += f"   📖 {movie['overview'][:100]}...\n\n"
        
        return response
    
    def extract_genres(self, genres_str):
        """Extract genre names from genres string"""
        try:
            import ast
            genres_list = ast.literal_eval(genres_str)
            return ', '.join([genre['name'] for genre in genres_list])
        except:
            return 'Various Genres'
    
    def get_help_response(self):
        """Comprehensive help response"""
        return """ **COMPLETE MOVIE EXPERT HELP** 

I can handle ANY movie request:

** BY GENRE:**
• Action, Comedy, Romance, Horror, Sci-Fi
• Drama, Fantasy, Animation, Family, Documentary

** BY MOOD:**
• Sad → Happy/Uplifting movies
• Happy → More joyful films  
• Bored → Exciting action
• Stressed → Relaxing comedies
• Romantic → Beautiful love stories

** BY OCCASION:**
• Birthday celebrations
• Date night movies
• Friends hangout
• Family time
• Alone time

** SEARCH:**
• "Movies like Inception"
• "Movies with Tom Cruise" 
• "Movies from 2020"
• "Popular movies"
• "Award-winning films"

** GENERAL:**
• Jokes, Facts, Stories
• Time, Date information
• Any conversation!

** MULTILINGUAL:**
• English and Bengali supported

**Just type ANYTHING naturally!**"""
    
    def get_joke(self):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything! ",
            "Why did the scarecrow win an award? He was outstanding in his field! ",
            "What do you call a fake noodle? An impasta! "
        ]
        return " **Movie Joke:**\n\n" + random.choice(jokes)
  
    def get_fact(self):
        facts = [
            "The first movie ever made was in 1888 - 'Roundhay Garden Scene' was only 2.11 seconds long! ",
            "The highest-grossing movie of all time is 'Avatar' with $2.8 billion! ",
            "The word 'cinema' comes from the Greek word 'kinema' meaning movement! "
        ]
        return " **Movie Fact:**\n\n" + random.choice(facts)
    
    def get_story(self):
        return " **Short Story:**\n\nOnce upon a time in Hollywood, a young filmmaker dreamed of creating the perfect movie... and with great stories and amazing visuals, they entertained millions! "