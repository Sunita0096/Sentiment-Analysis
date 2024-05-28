# Amazon Alexa Review - Sentiment Analysis

## Project Overview
This project involves analyzing Amazon Alexa reviews to predict whether the sentiment of a given review is positive or negative. We perform exploratory data analysis, preprocess the data, and build various classification models to achieve this goal. The models used include Random Forest, XGBoost, and Decision Tree classifiers.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Preprocessing](#preprocessing)
    - [Choosing TfidfVectorizer over CountVectorizer](#choosing-tfidfvectorizer-over-countvectorizer)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Conclusion](#conclusion)


## Introduction
Sentiment analysis helps understand the underlying sentiment expressed in textual data. This project focuses on analyzing user reviews of Amazon Alexa devices to determine if the sentiment is positive or negative. We use various machine learning models to classify the reviews.

## Dataset
The dataset used in this project is the Amazon Alexa reviews dataset. It contains the following columns:
- `rating`: The rating given by the user (1 to 5).
- `date`: The date of the review.
- `variation`: The variation of the product.
- `verified_reviews`: The textual review.
- `feedback`: The sentiment of the review (0 for negative, 1 for positive).

## Exploratory Data Analysis
We perform various analyses to understand the data better:
- **Distribution of Ratings**: Visualized using bar plots and pie charts.
- **Feedback Distribution**: Examined to understand the proportion of positive and negative reviews.
- **Variation Analysis**: Analyzed the different product variations and their ratings.
- **Review Length Analysis**: Explored the distribution of review lengths.

## Preprocessing
Text preprocessing steps include:
1. Replacing non-alphabet characters with spaces.
2. Converting text to lowercase and splitting into words.
3. Removing stopwords and stemming the remaining words.
4. Creating a bag-of-words model using TfidfVectorizer.

### Choosing TfidfVectorizer over CountVectorizer
Initially, we used `CountVectorizer` to create the bag-of-words model. However, we opted for `TfidfVectorizer` (Term Frequency-Inverse Document Frequency Vectorizer) for the following reasons:
- **Importance of Words**: `TfidfVectorizer` assigns a higher weight to words that are frequent in a document but not common across all documents. This helps in emphasizing important words while reducing the impact of common words like "the", "is", etc.
- **Feature Scaling**: TF-IDF naturally scales down the impact of the most frequent words, leading to a more balanced representation of the text.
- **Performance**: Models trained with TF-IDF vectors often perform better because the features are more informative.

## Modeling
We train three classifiers on the preprocessed data:
1. **Random Forest Classifier**
2. **XGBoost Classifier**
3. **Decision Tree Classifier**

Each model is evaluated for its performance using accuracy, confusion matrix, and cross-validation.

## Evaluation
The models are evaluated using the following metrics:
- **Accuracy**: Measured on both training and test sets.
- **Confusion Matrix**: To visualize the performance of the classifiers.
- **Cross-Validation**: Performed to ensure the model's robustness.

### Model Performance
#### Random Forest Classifier
- **Training Accuracy**: 99.46%
- **Testing Accuracy**: 94.18%

#### XGBoost Classifier
- **Training Accuracy**: 98.19%
- **Testing Accuracy**: 94.07%

#### Decision Tree Classifier
- **Training Accuracy**: 99.46%
- **Testing Accuracy**: 92.06%

## How to Run
To run this project, follow these steps:

### Prerequisites
- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `nltk`, `scikit-learn`, `xgboost`, `wordcloud`, `flask`

## Conclusion
This project demonstrates how to perform sentiment analysis on Amazon Alexa reviews using various machine learning models. By preprocessing the text data and employing different classifiers, we achieved high accuracy in predicting the sentiment of the reviews.


