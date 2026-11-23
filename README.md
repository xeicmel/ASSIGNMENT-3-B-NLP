# Politeness Detection in Malaysian TikTok Comments

## Project Overview
This project implements a text classification system to detect politeness in Malaysian TikTok comments. Comments are classified into two categories: Polite and Impolite.

This implementation focuses on traditional machine learning baseline models as part of the broader research objective of evaluating politeness detection approaches.

## Dataset
The dataset consists of 549 Malaysian TikTok comments labelled as Polite or Impolite.

- Polite: 305 comments
- Impolite: 244 comments

## Methodology
The implementation involves:
1. Dataset inspection and preparation
2. Train-test splitting (80:20)
3. TF-IDF feature extraction
4. Model training
5. Model evaluation and comparison
6. Prototype development

## Machine Learning Models
- Multinomial Naive Bayes
- Logistic Regression

## Feature Extraction
TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical features.

## Evaluation
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results
| Model | Accuracy |
|---|---:|
| Multinomial Naive Bayes | 89.09% |
| Logistic Regression | 85.45% |

Multinomial Naive Bayes achieved the higher accuracy in the conducted experiment.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF
- GitHub

## Project Scope
This repository contains the implementation of traditional machine learning baseline models for politeness classification. The implementation does not represent a completed evaluation of Large Language Models (LLMs).
