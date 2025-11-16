from flask import Flask, render_template, request, jsonify
from recommender import MovieRecommender
from nlp_model import CompleteMovieExpert
import os

app = Flask(__name__)

# Get the current directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build file paths dynamically
movies_path = os.path.join(BASE_DIR, 'data', 'movies.csv')
credits_path = os.path.join(BASE_DIR, 'data', 'credits.csv')

try:
    recommender = MovieRecommender(movies_path, credits_path)
    nlp_processor = CompleteMovieExpert(recommender)
    print("Complete Movie Expert initialized successfully!")
except Exception as e:
    print(f"Error initializing: {e}")
    recommender = None
    nlp_processor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not nlp_processor:
        return jsonify({'response': 'System initialization failed. Please check data files.'})

    user_message = request.json.get('message', '')

    if not user_message.strip():
        return jsonify({'response': 'Please enter a message.'})

    # Process with Complete NLP
    response = nlp_processor.process_query(user_message)

    return jsonify({'response': response})


# T: Render-compatible run

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
