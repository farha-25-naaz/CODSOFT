import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load data
def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if len(parts) < 4:
                continue
            plot = parts[3].strip()
            genres = [g.strip().lower() for g in parts[2].split(',')]
            data.append((plot, genres))
    return pd.DataFrame(data, columns=['plot', 'genres'])

train_df = load_data('train_data.txt')

# Text Preprocessing
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    if filtered:
        return ' '.join(filtered)
    else:
        return text

train_df['plot'] = train_df['plot'].apply(preprocess)
train_df = train_df[train_df['plot'].str.strip() != '']  # Remove empty plots

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,1))
X_train = vectorizer.fit_transform(train_df['plot'])

# Encode genres
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_df['genres'])

# Train model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, Y_train)

# --- User Input for Prediction ---
print("\n Try your own movie plot!")
plot_input = input("Enter movie plot description: ")
threshold_input = input("Enter probability threshold [default = 0.1]: ")
threshold = float(threshold_input) if threshold_input else 0.1

plot_cleaned = preprocess(plot_input)
X_user = vectorizer.transform([plot_cleaned])
probas = model.predict_proba(X_user)[0]

genre_probs = dict(zip(mlb.classes_, probas))

predicted_genres = [genre for genre, prob in genre_probs.items() if prob >= threshold]
print("\n🎬 Predicted Genres:", ', '.join(predicted_genres) if predicted_genres else "None")
