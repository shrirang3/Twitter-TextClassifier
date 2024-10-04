import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
# Load the classifier and vectorizer
clf = joblib.load("./Notebook/model_rf.pkl")
vectorizer = joblib.load("./Notebook/vectorizer.pkl")
explainer=joblib.load("C:/TMLC/PROJECT 10 (NLP)/Notebook/explainer.pkl")

# Function to perform sentiment prediction
def predict_sentiment(text_data):
    vectorized_text = vectorizer.transform([text_data])  # Vectorize input text
    prediction = clf.predict(vectorized_text)  # Predict using the loaded classifier
    return prediction

# Function to perform SHAP analysis and display the plots
def shap_analysis(text_input):
    # Vectorize the input text
    text_data = vectorizer.transform([text_input])    
    # Compute SHAP values
    shap_values = explainer.shap_values(text_data)
    
    # Generate SHAP force plot
    st.write("### SHAP Force Plot")
    shap.force_plot(explainer.expected_value[0], shap_values[0], text_data.toarray()[0], matplotlib=True)
    plt.savefig('force_plot.png', bbox_inches='tight')
    st.image('force_plot.png', caption="SHAP Force Plot", use_column_width=True)

    # Generate SHAP summary plot (bar)
    st.write("### SHAP Summary Plot")
    shap.initjs()
    words = vectorizer.get_feature_names_out()
    plt.figure()
    shap.summary_plot(shap_values[1], text_data, feature_names=words, plot_type="bar")
    
    # Display the SHAP summary plot in Streamlit
    st.pyplot(plt)

    # Clean up the Matplotlib plot to avoid overlap
    plt.clf()
