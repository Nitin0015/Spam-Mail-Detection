# **Spam Detection**

This repository contains a Jupyter Notebook that demonstrates how to classify SMS messages as spam or non-spam (ham) using machine learning techniques. The dataset used in this project includes labeled SMS messages.

---

## **Overview**

Spam detection is a crucial task in natural language processing (NLP) to filter out unwanted or malicious messages. This project uses **Logistic Regression** to classify SMS messages into two categories: spam or ham. Text preprocessing techniques like TF-IDF vectorization are applied to convert text data into numerical features suitable for machine learning models.

The dataset includes SMS messages and their corresponding labels (`spam` or `ham`).

---

## **Dataset**

- **Source**: The dataset appears to be related to publicly available SMS spam detection datasets.
- **Features**:
  - `Category`: Indicates whether the message is `spam` or `ham`.
  - `Message`: The content of the SMS message.

---

## **Project Workflow**

1. **Data Loading**:
   - The dataset (`mail_data.csv`) is loaded into a Pandas DataFrame.
2. **Exploratory Data Analysis (EDA)**:
   - Summary statistics and visualizations are generated to understand the distribution of spam and ham messages.
3. **Data Preprocessing**:
   - Text data is cleaned and transformed using TF-IDF Vectorization to convert it into numerical format.
4. **Model Training**:
   - A Logistic Regression model is trained on the transformed text data.
   - The dataset is split into training and testing sets using `train_test_split`.
5. **Model Evaluation**:
   - Accuracy score is calculated to evaluate the model's performance.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

You can install these dependencies using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/SpamDetection.git
   cd SpamDetection
   ```

2. Ensure that the dataset file (`mail_data.csv`) is in the same directory as the notebook.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Spam-Detect.ipynb
   ```

4. Run all cells in the notebook to execute the code.

---

## **Results**

The Logistic Regression model predicts whether an SMS message is spam or ham. The accuracy score indicates how well the model performs in classifying messages. Experimenting with other machine learning models or advanced NLP techniques can further improve the model's performance.

---

## **Acknowledgments**

- The dataset was sourced from publicly available SMS spam detection datasets.
- Special thanks to Scikit-learn for providing robust machine learning tools.

---
