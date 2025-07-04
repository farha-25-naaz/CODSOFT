# Spam Detection using TF-IDF and Logistic Regression
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words - {'call', 'free', 'won', 'congratulations'}

# Load the Dataset
df = pd.read_csv('spam.csv', encoding='latin-1',usecols=[0,1])
df.columns = ['label', 'text']  

# Clean the Text
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in custom_stopwords]
    return ' '.join(words)

df['text_clean'] = df['text'].apply(clean_text)

# Step 3: Vectorization using TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1, 2))
X = tfidf.fit_transform(df['text_clean'])  # Features
y = df['label'].map({'ham': 0, 'spam': 1})  # Labels

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Classifier
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict on New Message
def predict_message(msg, threshold=0.5):
    msg_clean = clean_text(msg)
    msg_vector = tfidf.transform([msg_clean])
    proba = model.predict_proba(msg_vector)[0][1]
    if proba > threshold:
        return f"SPAM ❌ (prob: {proba:.2f})"
    else:
        return f"HAM ✅ (prob: {proba:.2f})"

# Try a sample message
sample =" 🎉 Congratulations! You've won a ₹10,00,000 lottery prize! Click here to claim now: http://fake-lottery-prize.xyz 🎁"
print("Prediction:", predict_message(sample, threshold=0.4))
