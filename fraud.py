import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

print("🚀 Script started...")

try:
    df = pd.read_csv('fraudTest.csv', encoding='ISO-8859-1', on_bad_lines='skip')
    print(f"Loaded dataset with {len(df)} rows")
except Exception as e:
    print("❌ Error loading file:", e)
    exit()

# Sample 200,000 rows
sample_size = 300000
df = df.sample(n=sample_size, random_state=42)
print(f"Sampled {sample_size} rows for training.")

if 'is_fraud' not in df.columns:
    print("❌ 'is_fraud' column not found. Columns present:", df.columns.tolist())
    exit()

x = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Datetime handling
if 'trans_date_trans_time' in x.columns:
    x['trans_date_trans_time'] = pd.to_datetime(x['trans_date_trans_time'], errors='coerce')
    x['hour'] = x['trans_date_trans_time'].dt.hour
    x['day'] = x['trans_date_trans_time'].dt.day
    x['weekday'] = x['trans_date_trans_time'].dt.weekday
    x = x.drop('trans_date_trans_time', axis=1)

# Drop irrelevant columns
x = x.drop(columns=[col for col in ['cc_num', 'trans_num', 'unix_time', 'dob', 'first', 'last', 'street'] if col in x.columns])

# Encode categorical
x = x.fillna("unknown")
categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
x = pd.get_dummies(x, columns=[col for col in categorical_cols if col in x.columns], drop_first=True)

# Ensure numeric
x = x.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"Using {x.shape[1]} features after encoding.")

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train
print("Training model with 100 trees...")
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(x_train, y_train)

# Predict
print("Predicting on test set...")
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\n🧮 Confusion Matrix:")
print(conf_matrix)

fraud_pred_count = sum(y_pred == 1)
legit_pred_count = sum(y_pred == 0)

print(f"\n💳 Predicted Fraudulent Transactions: {fraud_pred_count}")
print(f"✅ Predicted Legitimate Transactions: {legit_pred_count}")
print(f"📦 Processed {len(df)} rows.")

# Terminal-style bar chart
def print_bar(label, count, total, bar_len=50):
    filled_len = int(bar_len * count / total)
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    print(f"{label:<12}: |{bar}| {count}")

print("\n📈 Terminal-style Transaction Prediction Chart:")
total = fraud_pred_count + legit_pred_count
print_bar("Legitimate", legit_pred_count, total)
print_bar("Fraudulent", fraud_pred_count, total)

# Matplotlib horizontal bar chart
labels = ['Legitimate', 'Fraudulent']
counts = [legit_pred_count, fraud_pred_count]

plt.figure(figsize=(6, 3))
plt.barh(labels, counts, color=['#90ee90', '#ff9999'])
plt.xlabel('Count')
plt.title('Predicted Transactions')
for i, v in enumerate(counts):
    plt.text(v + 10, i, str(v), va='center')
plt.tight_layout()
plt.show()

input("\nPress Enter to exit...")
