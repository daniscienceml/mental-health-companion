from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
import pickle
import re
import os

app = Flask(__name__)

# Load Model
# BASE = r'C:\Users\User\Desktop\mental_health_companion'
import os

BASE = r'C:\Users\User\Desktop\mental_health_companion'
MODELS = os.path.join(BASE, 'models')

with open(os.path.join(MODELS, 'mental_health_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(MODELS, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

# Keywords
CRISIS_KEYWORDS = [
    'hate life', 'no reason to live', 'end my life',
    'kill myself', 'want to die', 'cant go on',
    'hopeless', 'worthless', 'nobody cares',
    'very sad', 'deeply sad', 'feel empty',
    'depressed', 'depression', 'no hope', 'give up',
    'lonely', 'alone', 'no one cares',
    'very worried', 'very anxious', 'panic',
    'cant stop thinking', 'overthinking',
    'cant sleep', "can't sleep", 'no sleep',
    'insomnia', 'not sleeping',
    'very tired', 'exhausted', 'no energy',
    'fatigue', 'drained', 'worn out',
    'no appetite', 'not eating',
    'cry all day', 'cant stop crying',
    'struggling', 'suffering', 'cant cope',
    'overwhelmed', 'stressed', 'burned out'
]

# Coping Strategies
COPING = {
    'mental_health': [
        "گہری سانس لیں — 4 سیکنڈ اندر، 4 سیکنڈ باہر",
        "کسی قریبی دوست یا گھر والے سے بات کریں",
        "باہر 10 منٹ چہل قدمی کریں — دھوپ مددگار ہے",
        "اپنی 3 اچھی باتیں ایک کاغذ پر لکھیں",
        "پانی پیئں اور کچھ ہلکا کھائیں"
    ],
    'uncertain': [
        "اپنے جذبات کو سمجھنے کی کوشش کریں",
        "ڈائری میں اپنے خیالات لکھیں",
        "کسی قابل اعتماد شخص سے بات کریں"
    ],
    'normal': [
        "خوش رہیں اور اپنا خیال رکھیں! 😊",
        "آج کے اچھے لمحوں کا شکریہ ادا کریں",
        "اپنی خوشی کو دوسروں کے ساتھ بانٹیں"
    ]
}

# Translate
def translate_to_english(text):
    try:
        return GoogleTranslator(
            source='auto',
            target='english'
        ).translate(text)
    except:
        return text

# Predict
def predict(text):
    english = translate_to_english(text)
    clean = english.lower()
    clean = re.sub(r'[^a-zA-Z\s]', '', clean)

    keyword_match = any(k in english.lower() for k in CRISIS_KEYWORDS)

    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    confidence = round(max(probability) * 100, 1)

    if keyword_match or prediction == 1:
        if confidence < 65 and not keyword_match:
            category = 'uncertain'
            urdu_result = "آپ کی کیفیت واضح نہیں — مزید بتائیں"
        else:
            category = 'mental_health'
            urdu_result = "ذہنی صحت کا مسئلہ محسوس ہوا"
    else:
        category = 'normal'
        urdu_result = "آپ ٹھیک لگ رہے ہیں"

    return {
        'urdu_result': urdu_result,
        'category': category,
        'confidence': confidence,
        'translated': english,
        'coping': COPING[category]
    }

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Please enter text'})
    result = predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)