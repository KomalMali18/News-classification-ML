# 📰 News Classification using Machine Learning  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Model-orange)
![Status](https://img.shields.io/badge/Project%20Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📖 Overview  
This project focuses on **automatically classifying news articles** into various categories using **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques.  
It processes raw news data, cleans and transforms it into numerical features, trains classification models, and evaluates their accuracy with visual results.

---

## 🧰 Tech Stack  
| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 🐍 |
| **Libraries** | pandas, numpy, scikit-learn, matplotlib, seaborn |
| **Techniques** | NLP, TF-IDF Vectorization, Text Preprocessing |
| **Models** | Logistic Regression, Multinomial Naive Bayes |

---

## 🗂️ Dataset Details  
| Feature | Description |
|----------|-------------|
| **Title** | The headline of the news article |
| **Text / Body** | Full content of the article |
| **Category** | Target label (e.g., Sports, Politics, Business, Technology, etc.) |
| **Format** | CSV file |

🧾 The dataset is preprocessed to remove punctuation, convert text to lowercase, and eliminate stopwords before model training.

---

## ⚙️ Workflow  
### 🔹 Step 1: Data Preprocessing  
- Cleaned and tokenized the text  
- Removed stopwords and punctuations  
- Converted text to numerical form using **TF-IDF Vectorizer**

### 🔹 Step 2: Model Training  
- Implemented and compared ML algorithms:  
  - **Multinomial Naive Bayes**  
  - **Logistic Regression**

### 🔹 Step 3: Evaluation  
- Evaluated using **Accuracy**, **Confusion Matrix**, and **Classification Report**  
- Visualized results with **Matplotlib** and **Seaborn**

---

## 📊 Results & Insights  
| Metric | Value |
|---------|-------|
| **Best Model Accuracy** | **~53.46%** |
| **Output Visuals** | Category distribution plots, text length analysis, confusion matrix heatmap |

> 🧠 The Naive Bayes model performed well for balanced categories, while Logistic Regression provided better precision on frequent labels.

---

## 🚀 Future Improvements  
- Upgrade to **Deep Learning models** (LSTM / BERT)  
- Add **real-time news classification web app** using Flask or Streamlit  
- Enrich dataset with more samples and diverse sources  
- Implement **hyperparameter tuning** for improved accuracy  

---

## 📸 Sample Outputs  
- ✅ Category Distribution Plot  
- ✅ Text Length Analysis Graph  
- ✅ Confusion Matrix Heatmap  

---

## 👩‍💻 Author  
**👤 Komal Mali**  
🔗 [GitHub Profile](https://github.com/KomalMali18)  

---

## 📜 License  
This project is licensed under the **MIT License** – feel free to use and modify it.

---

⭐ **If you found this project helpful, don’t forget to give it a star!** ⭐
