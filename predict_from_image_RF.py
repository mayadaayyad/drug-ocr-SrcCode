import cv2
import pytesseract
import joblib
from rapidfuzz import process, fuzz

# Load the trained classification model and the TF-IDF vectorizer
model = joblib.load(r"C:\Users\NOUR SOFT\Desktop\Final_Graduation\rf_model_weighted.pkl")
vectorizer = joblib.load(r"C:\Users\NOUR SOFT\Desktop\Final_Graduation\rf_wvectorizer.pkl")


# Load the reference list of valid drug names from a text file
def load_drug_list_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

drug_list = load_drug_list_from_file(r"C:\Users\NOUR SOFT\Desktop\Final_Graduation\drugs.txt")


# Extract text from a given image using Tesseract OCR
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# Clean and tokenize the OCR-extracted text
def extract_words_from_text(text):
    words = text.split()
    return [w.lower() for w in words if w.isalpha() and len(w) > 3]


# Perform fuzzy matching against the drug list
def smart_fuzzy_match(word, drug_list):
    candidates = [d for d in drug_list if abs(len(d) - len(word)) <= 3]
    if not candidates:
        return None, 0.0
    # Use three different scorers and select the best match
    match = max(
        [
            process.extractOne(word, candidates, scorer=fuzz.ratio),
            process.extractOne(word, candidates, scorer=fuzz.token_sort_ratio),
            process.extractOne(word, candidates, scorer=fuzz.token_set_ratio)
        ],
        key=lambda x: x[1] if x else 0
    )
    return match[0], match[1] if match else (None, 0.0)


# Use the trained ML model to verify if a word is a drug
def verify_with_model_prediction(word, vectorizer, model):
    X = vectorizer.transform([word])
    prediction = model.predict(X)[0]
    return prediction == 1

def extract_and_validate_drugs(image_path):
    text = extract_text_from_image(image_path)
    extracted_words = extract_words_from_text(text)

    final_drugs = []
    for word in extracted_words:
        match, score = smart_fuzzy_match(word, drug_list)
        if match:
            predicted = verify_with_model_prediction(match, vectorizer, model)
            print(f"🌀 Word: {word} → Fuzzy Match: {match} ({score}) → Model Prediction: {predicted}")

            if predicted or score >= 80:
                final_drugs.append(match)

    return extracted_words, final_drugs

# ---- MAIN ----
if __name__ == "__main__":
    image_path = r"C:\Users\NOUR SOFT\Desktop\\Final_Graduation\486.jpg"

    extracted_words, final_drugs = extract_and_validate_drugs(image_path)

    print("\n📜 OCR Extracted Words:")
    for w in extracted_words:
        print("-", w)

    print("\n✅ Final Predicted Drug Names (after fuzzy + model):")
    for d in final_drugs:
        print("-", d)
