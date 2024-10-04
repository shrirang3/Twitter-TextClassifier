import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
from processing import predict_sentiment, shap_analysis


# Set up the page title and layout
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")
st.title("Twitter Sentiment Analyzer")
st.subheader("Classify sentiment from custom text or CSV file")

# Sidebar for selecting input method
st.sidebar.subheader("Choose Input Method")
input_type = st.sidebar.radio("Input type", ("Text Input", "CSV Upload"))

if input_type == "Text Input":
    # Text input section
    text_input = st.text_area("Enter your tweet here", placeholder="Type your tweet...")

    if st.button("Analyze Sentiment"):
        if text_input.strip():
            sentiment = predict_sentiment(text_input)
            prediction= sentiment[0]
            if prediction == 1:
                st.write(f"Predicted Sentiment: Hate Speech")
            else:
                st.write(f"Predicted Sentiment: Non-Hate Speech")
            
            # Show SHAP analysis
            st.write("SHAP Analysis:")
            shap_analysis(text_input)
        else:
            st.warning("Please enter some text to analyze")

elif input_type == "CSV Upload":
    # CSV upload section
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())  # Show a preview of the uploaded CSV file

        if "Text" in df.columns:
            if st.button("Analyze CSV"):
                predictions = df["Text"].apply(predict_sentiment)  # Apply predictions to CSV text
                df['Sentiment'] = predictions
                st.write("Sentiment analysis completed!")
                st.write(df[['Text', 'Sentiment']])

                # Optionally download the results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download result as CSV", data=csv, file_name="sentiment_results.csv")
        else:
            st.warning("The CSV file should contain a 'Text' column.")
else:
    st.info("Please select an input method from the sidebar")
