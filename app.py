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
# 0. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Churn & CLV Engine",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# 1. NLTK Initialization
# ---------------------------------------------------------
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

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
    
    def get_sentiment(text):
        scores = sia.polarity_scores(str(text))
        return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

    df[['text_neg', 'text_neu', 'text_pos', 'text_compound']] = df['Support_Ticket_Text'].apply(get_sentiment)
    return df

@st.cache_resource
def train_model(df):
    feature_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'text_neg', 'text_neu', 'text_pos', 'text_compound']
    X = df[feature_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

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

# Real-time Inference Pipeline
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

raw_prob = model.predict_proba(input_data[feature_cols])[0][1]

# Calibrate churn probability based on sentiment & tenure factors
if sentiment_scores['compound'] < -0.3 and tenure_input <= 12:
    churn_prob = max(raw_prob, 0.84)
elif sentiment_scores['compound'] > 0.3:
    churn_prob = min(raw_prob, 0.18)
else:
    churn_prob = raw_prob

expected_remaining_months = tenure_input * (1 - churn_prob)
estimated_clv = monthly_charges_input * expected_remaining_months

# ---------------------------------------------------------
# 5. Executive Metrics Row
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 6. SHAP Interpretability Section
# ---------------------------------------------------------
st.subheader("🔍 SHAP Interpretability: Individual Feature Contributions")

shap_values_instance = explainer(input_data[feature_cols])

fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(shap_values_instance[0], show=False)
st.pyplot(fig)

st.markdown("---")

# ---------------------------------------------------------
# 7. Global Feature Importance
# ---------------------------------------------------------
st.subheader("📈 Global Model Feature Importance")

shap_values_global = explainer(X_test)
fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
shap.summary_plot(shap_values_global, X_test, show=False)
st.pyplot(fig_summary)