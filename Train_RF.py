import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load the prepared training dataset
df = pd.read_csv(r"C:\Users\NOUR SOFT\Desktop\Final_Graduation\prepared_training_data.csv")

# 2. Drop any rows with missing values in 'word' or 'is_drug' columns
df.dropna(subset=["word", "is_drug"], inplace=True)

# 3. Convert words into numerical vectors using TF-IDF (with unigrams & bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = vectorizer.fit_transform(df["word"].astype(str))
y = df["is_drug"]

# 4. Drug words (label = 1) are given higher importance (weight = 2.0) و Non-drug words (label = 0) are given lower importance (weight = 0.5)
sample_weights = df["is_drug"].apply(lambda x: 2.0 if x == 1 else 0.5)

# 5. Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weights, test_size=0.2, stratify=y, random_state=42
)

# 6. Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train, sample_weight=sw_train)


# 7. Evaluate model performance on the training set
y_train_pred = rf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"📈 Training Accuracy: {train_acc*100:.2f}%")


# 8. Evaluate model performance on the test set
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy on Test Set: {acc*100:.2f}%\n")
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))


joblib.dump(rf, r"C:\Users\NOUR SOFT\Desktop\Final_Graduation\rf_model_weighted.pkl")
joblib.dump(vectorizer, r"C:\Users\NOUR SOFT\Desktop\Final_Graduation\rf_wvectorizer.pkl")
