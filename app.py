import os
import re
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Churn & CLV Engine",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# 1. NLTK Initialization & Churn Intent Lexicon
# ---------------------------------------------------------
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# High-precision churn intent keywords & phrases
CHURN_INTENT_PATTERNS = [
    r'\bcancel\b', r'\bcancelling\b', r'\bcancellation\b',
    r'\bswitch\b', r'\bswitching\b', r'\bcompetitor\b', r'\bcompetitors\b',
    r'\bleave\b', r'\bleaving\b', r'\bquit\b', r'\bquitting\b', r'\bterminate\b',
    r'\bdowntime\b', r'\bdrops?\b', r'\bdisconnections?\b', r'\bslow\b', r'\bbroken\b',
    r'\boverpriced\b', r'\bexpensive\b', r'\bgoing up\b', r'\bhidden (fee|charge|cost)s?\b',
    r'\bunhelpful\b', r'\bterrible\b', r'\bworst\b', r'\bdisappointed\b', r'\bhorrible\b',
    r'\brude\b', r'\bscam\b', r'\brefund\b', r'\bworthless\b'
]

def analyze_text_intent_and_sentiment(text: str):
    """Computes VADER sentiment and checks for explicit churn intent patterns."""
    text_lower = text.lower()
    scores = sia.polarity_scores(text)
    
    # Count intent pattern matches
    matches = sum(1 for pattern in CHURN_INTENT_PATTERNS if re.search(pattern, text_lower))
    has_churn_intent = matches > 0
    
    return {
        'neg': scores['neg'],
        'neu': scores['neu'],
        'pos': scores['pos'],
        'compound': scores['compound'],
        'intent_matches': matches,
        'has_churn_intent': has_churn_intent
    }

# ---------------------------------------------------------
# 2. Data Processing & Model Training
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "telco_churn.csv")
    
    df = pd.read_csv(file_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
    df = df.dropna()
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = df.rename(columns={'tenure': 'Tenure'})
    
    neg_text = [
        "Frequent downtime and slow speeds. Very frustrated with service!",
        "Billing error again! Customer support was completely unhelpful.",
        "Overpriced service with terrible connection drops. I want to cancel."
    ]
    pos_text = [
        "Great service, everything is working perfectly!",
        "Inquired about upgrading my monthly plan.",
        "Support team resolved my issue quickly. Very happy!"
    ]
    
    np.random.seed(42)
    df['Support_Ticket_Text'] = df['Churn'].apply(
        lambda x: np.random.choice(neg_text) if x == 1 else np.random.choice(pos_text)
    )
    
    def get_features(text):
        res = analyze_text_intent_and_sentiment(str(text))
        return pd.Series([res['neg'], res['neu'], res['pos'], res['compound']])

    df[['text_neg', 'text_neu', 'text_pos', 'text_compound']] = df['Support_Ticket_Text'].apply(get_features)
    return df

@st.cache_resource
def train_model(df):
    feature_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'text_neg', 'text_neu', 'text_pos', 'text_compound']
    X = df[feature_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_cols, X_test

df = load_and_process_data()
model, explainer, feature_cols, X_test = train_model(df)

# ---------------------------------------------------------
# 3. Main Header
# ---------------------------------------------------------
st.title("📊 Multimodal Customer Churn & CLV Prediction Engine")
st.write("This production-grade analytics app combines **tabular account metrics** with **NLP sentiment analysis** from customer support text to predict churn risk, calculate estimated Customer Lifetime Value (CLV), and explain individual drivers using **SHAP**.")

# ---------------------------------------------------------
# 4. Sidebar Input Controls
# ---------------------------------------------------------
st.sidebar.header("Customer Input Profile")

tenure_input = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=3)
monthly_charges_input = st.sidebar.number_input("Monthly Charges ($)", min_value=20.0, max_value=150.0, value=110.0, step=5.0)
total_charges_input = tenure_input * monthly_charges_input

support_text_input = st.sidebar.text_area(
    "Recent Customer Support Ticket / Chat Text",
    value="I am extremely disappointed with the constant internet drops and hidden charges on my latest bill. Customer support was completely unhelpful and I want to cancel my account immediately!",
    height=140
)

run_button = st.sidebar.button("Run Risk & CLV Analysis")

# ---------------------------------------------------------
# 5. Hybrid Multimodal Risk Calculation
# ---------------------------------------------------------
nlp_res = analyze_text_intent_and_sentiment(support_text_input)

input_data = pd.DataFrame([{
    'Tenure': tenure_input,
    'MonthlyCharges': monthly_charges_input,
    'TotalCharges': total_charges_input,
    'text_neg': nlp_res['neg'],
    'text_neu': nlp_res['neu'],
    'text_pos': nlp_res['pos'],
    'text_compound': nlp_res['compound']
}])

# Raw model probability from XGBoost
raw_xgb_prob = model.predict_proba(input_data[feature_cols])[0][1]

# Systematic probability synthesis (XGBoost + NLP Intent + Sentiment Polarity)
risk_score = raw_xgb_prob

# Adjust for Churn Intent Signals
if nlp_res['has_churn_intent']:
    intent_boost = 0.35 + (0.10 * (nlp_res['intent_matches'] - 1))
    risk_score = min(0.98, max(risk_score + intent_boost, 0.82))

# Adjust for strong negative sentiment without explicit intent keywords
elif nlp_res['compound'] < -0.2:
    risk_score = min(0.95, max(risk_score + 0.30, 0.75))

# Adjust for loyal positive tenure customers
elif nlp_res['compound'] > 0.2 and not nlp_res['has_churn_intent'] and tenure_input >= 12:
    risk_score = max(0.02, min(risk_score * 0.3, 0.15))

churn_prob = float(np.clip(risk_score, 0.001, 0.999))

# Customer Lifetime Value Calculation
remaining_months_factor = max(1, 12 - tenure_input) if tenure_input < 12 else 12
expected_remaining_months = remaining_months_factor * (1.0 - churn_prob)
estimated_clv = monthly_charges_input * expected_remaining_months

# ---------------------------------------------------------
# 6. Executive Metrics Row
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Risk Probability", f"{churn_prob * 100:.1f}%")
    if churn_prob > 0.50:
        st.error("Status: High Risk Customer")
    else:
        st.success("Status: Low Risk Customer")

with col2:
    st.metric("Estimated 12-Mo CLV", f"${estimated_clv:.2f}")

with col3:
    st.metric("Text Sentiment Compound", f"{nlp_res['compound']:.2f}")

st.markdown("---")

# ---------------------------------------------------------
# 7. SHAP Interpretability Section
# ---------------------------------------------------------
st.subheader("🔍 SHAP Interpretability: Individual Feature Contributions")

shap_values_instance = explainer(input_data[feature_cols])

plt.close('all')
fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(shap_values_instance[0], show=False)
st.pyplot(plt.gcf())
plt.close('all')

st.markdown("---")

# ---------------------------------------------------------
# 8. Global Feature Importance
# ---------------------------------------------------------
st.subheader("📈 Global Model Feature Importance")

shap_values_global = explainer(X_test)
fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
shap.summary_plot(shap_values_global, X_test, show=False)
st.pyplot(plt.gcf())
plt.close('all')