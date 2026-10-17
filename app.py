import os
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
# 0. NLTK Initialization
# ---------------------------------------------------------
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ---------------------------------------------------------
# 1. Real Kaggle Dataset Loading & Multimodal Preprocessing
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data():
    # Automatically locate telco_churn.csv in the same directory as app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "telco_churn.csv")
    
    df = pd.read_csv(file_path)
    
    # Clean data (Kaggle's TotalCharges contains blank spaces)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
    df = df.dropna()
    
    # Format target variable and column names
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = df.rename(columns={'tenure': 'Tenure'})
    
    # Simulate support ticket text logs
    neg_text = [
        "Frequent downtime and slow speeds. Very frustrated!",
        "Billing error again! Customer support was unhelpful.",
        "Overpriced service with terrible connection drops."
    ]
    pos_text = [
        "Great service, everything is working perfectly!",
        "Inquired about upgrading my monthly plan.",
        "Support team resolved my issue quickly."
    ]
    
    np.random.seed(42)
    df['Support_Ticket_Text'] = df['Churn'].apply(
        lambda x: np.random.choice(neg_text) if x == 1 else np.random.choice(pos_text)
    )
    
    # Extract Sentiment Scores
    def get_sentiment(text):
        scores = sia.polarity_scores(str(text))
        return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

    df[['text_neg', 'text_neu', 'text_pos', 'text_compound']] = df['Support_Ticket_Text'].apply(get_sentiment)
    
    return df

# ---------------------------------------------------------
# 2. Model Training & SHAP Setup
# ---------------------------------------------------------
@st.cache_resource
def train_model(df):
    feature_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'text_neg', 'text_neu', 'text_pos', 'text_compound']
    X = df[feature_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_cols, X_test

# Load data and train model
df = load_and_process_data()
model, explainer, feature_cols, X_test = train_model(df)

# ---------------------------------------------------------
# 3. Streamlit Interface
# ---------------------------------------------------------
st.set_page_config(page_title="Multimodal Churn & CLV Engine", layout="wide")

st.title("📊 Multimodal Customer Churn & CLV Prediction Engine")
st.markdown("""
This production-grade analytics app combines **tabular account metrics** with **NLP sentiment analysis** from customer support text to predict churn risk, calculate estimated Customer Lifetime Value (CLV), and explain individual drivers using **SHAP**.
""")

st.sidebar.header("Customer Input Profile")

tenure_input = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges_input = st.sidebar.number_input("Monthly Charges ($)", min_value=20.0, max_value=150.0, value=75.0)
total_charges_input = tenure_input * monthly_charges_input

support_text_input = st.sidebar.text_area(
    "Recent Customer Support Ticket / Chat Text",
    value="My internet connection keeps dropping during work hours and customer service was unhelpful."
)

if st.sidebar.button("Run Risk & CLV Analysis"):
    sentiment_scores = sia.polarity_scores(support_text_input)
    
    input_data = pd.DataFrame([{
        'Tenure': tenure_input,
        'MonthlyCharges': monthly_charges_input,
        'TotalCharges': total_charges_input,
        'text_neg': sentiment_scores['neg'],
        'text_neu': sentiment_scores['neu'],
        'text_pos': sentiment_scores['pos'],
        'text_compound': sentiment_scores['compound']
    }])

    churn_prob = model.predict_proba(input_data[feature_cols])[0][1]
    
    expected_remaining_months = tenure_input * (1 - churn_prob)
    estimated_clv = monthly_charges_input * expected_remaining_months

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Churn Risk Probability", f"{churn_prob * 100:.1f}%")
        if churn_prob > 0.5:
            st.error("Status: High Risk Customer")
        else:
            st.success("Status: Low Risk Customer")

    with col2:
        st.metric("Estimated 12-Mo CLV", f"${estimated_clv:.2f}")

    with col3:
        st.metric("Text Sentiment Compound", f"{sentiment_scores['compound']:.2f}")

    st.markdown("---")
    st.subheader("🔍 SHAP Interpretability: Individual Feature Contributions")

    shap_values_instance = explainer(input_data[feature_cols])

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values_instance[0], show=False)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📈 Global Model Feature Importance (SHAP Summary)")
    
    shap_values_global = explainer(X_test)
    fig_summary, ax_summary = plt.subplots(figsize=(8, 4))
    shap.summary_plot(shap_values_global, X_test, show=False)
    st.pyplot(fig_summary)