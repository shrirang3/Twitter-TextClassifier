
# Twitter Text Classifier

The project performs a  step-by-step NLP approach on twitter_parsed_tweets to identify the toxicity of the tweet, and classify the tweet in a binary value. 


## Building the Repo locally:

Install required libraries:

```bash
  pip install -r requirements.txt
```

Run front-end script:

```bash
  streamlit run app.py
```



## Project Pipeline

This project aims to classify text data using machine learning techniques. The following steps outline the process:

### 1. **Data Preprocessing**
   - **Removing Numbers**: All numeric characters were removed to focus on textual information.
   - **Removing Hyperlinks**: URLs and hyperlinks were identified and discarded from the text data.
   - **Removing Tags**: HTML and other unwanted tags were stripped out of the data.
   - **Removing Emojis**: Emojis were excluded to ensure uniformity in the textual content.
   - **Lowercase Conversion**: All text was converted to lowercase to ensure case uniformity and eliminate case sensitivity during analysis.
   - **Removing Stopwords**: Common stopwords (e.g., "the", "is", "in") were removed to focus on meaningful words.
   - **Lemmatization**: The text was lemmatized to reduce words to their base or dictionary form (e.g., "running" becomes "run").

### 2. **Data Visualization**
   - Visualization techniques, such as word clouds and frequency distribution plots, were used to analyze the most frequent words and overall distribution of the text data.

### 3. **Feature Extraction**
   - **TF-IDF Word Embeddings**: Term Frequency-Inverse Document Frequency (TF-IDF) was used to convert the text data into numerical vectors that could be used for machine learning models.

### 4. **Data Splitting**
   - The processed data was split into training and validation sets to evaluate model performance. A typical train-validation split ratio was 80-20.

### 5. **Model Training**
   - The training dataset was used to build a machine learning model. Different models were evaluated, and Random-Forest Classifier was considered the best.
