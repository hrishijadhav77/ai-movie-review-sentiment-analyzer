import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# -----------------------------------
# Load DistilBERT Model
# -----------------------------------
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "models/distilbert_final"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "models/distilbert_final"
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# -----------------------------------
# Session State
# -----------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "review_text" not in st.session_state:
    st.session_state.review_text = ""


# -----------------------------------
# Prediction Function
# -----------------------------------
def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    label = (
        "Positive"
        if prediction.item() == 1
        else "Negative"
    )

    return (
        label,
        confidence.item()
    )


# -----------------------------------
# Streamlit Page Config
# -----------------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.success(
    "✅ DistilBERT Model Loaded Successfully"
)


# -----------------------------------
# Title
# -----------------------------------
st.title(
    "🎬 Movie Review Sentiment Analyzer"
)

st.write(
    """
    Enter a movie review and let DistilBERT
    predict whether it is Positive or Negative.
    """
)


# -----------------------------------
# Project Statistics Dashboard
# -----------------------------------

st.subheader(
    "📈 Project Statistics"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Dataset Size",
        "49,582"
    )

with col2:

    st.metric(
        "Model",
        "DistilBERT",
        "Transformer"
    )

col3, col4 = st.columns(2)

with col3:

    st.metric(
        "Accuracy",
        "91.46%"
    )

with col4:

    st.metric(
        "Training Samples",
        "39,665"
    )

st.write("---")



# -----------------------------------
# Example Reviews
# -----------------------------------
st.subheader(
    "Try Example Reviews"
)

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "😊 Positive Example"
    ):

        st.session_state.review_text = """
This movie was absolutely fantastic.
The acting was brilliant and the story
kept me engaged from start to finish.
Highly recommended.
"""

with col2:

    if st.button(
        "☹️ Negative Example"
    ):

        st.session_state.review_text = """
This movie was extremely boring.
The story was weak and the acting was terrible.
I would not recommend it.
"""


# -----------------------------------
# Review Input
# -----------------------------------
review = st.text_area(
    "Enter Review",
    value=st.session_state.review_text,
    height=150
)


# -----------------------------------
# Analyze Button
# -----------------------------------
if st.button(
    "Analyze Sentiment"
):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        label, confidence = predict_sentiment(
            review
        )

        # Save Prediction History
        st.session_state.history.append(
            {
                "Review": review[:80] + "...",
                "Sentiment": label,
                "Confidence (%)": round(
                    confidence * 100,
                    2
                )
            }
        )

        st.write("---")

        st.subheader(
            "Prediction Result"
        )

        # Professional Sentiment Card
        if label == "Positive":

            st.success(
                "🟢 POSITIVE REVIEW"
            )

        else:

            st.error(
                "🔴 NEGATIVE REVIEW"
            )

        # Confidence Score
        st.metric(
            "Confidence Score",
            f"{confidence * 100:.2f}%"
        )

        # Progress Bar
        st.progress(
            float(confidence)
        )


# -----------------------------------
# Model Information
# -----------------------------------
st.write("---")

st.subheader(
    "Model Information"
)

st.info(
    """
**Model:** DistilBERT

**Dataset:** IMDB Movie Reviews

**Classes:** Positive / Negative

**Accuracy:** 91.46%
"""
)


# -----------------------------------
# Prediction History
# -----------------------------------
if len(
    st.session_state.history
) > 0:

    st.write("---")

    st.subheader(
        "Prediction History"
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )


# -----------------------------------
# Analytics Sidebar
# -----------------------------------
st.sidebar.title(
    "📊 Analytics"
)

total_predictions = len(
    st.session_state.history
)

positive_count = sum(
    1
    for item in st.session_state.history
    if item["Sentiment"] == "Positive"
)

negative_count = sum(
    1
    for item in st.session_state.history
    if item["Sentiment"] == "Negative"
)

st.sidebar.metric(
    "Total Predictions",
    total_predictions
)

st.sidebar.metric(
    "Positive Reviews",
    positive_count
)

st.sidebar.metric(
    "Negative Reviews",
    negative_count
)


# -----------------------------------
# Footer
# -----------------------------------
st.write("---")

st.caption(
    "Built by Hrishikesh Jadhav"
)

st.caption(
    "DistilBERT Sentiment Analysis Project"
)