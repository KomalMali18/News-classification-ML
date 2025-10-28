# ==========================================
# 📰 Clean & Beautiful Version
# Vertical Bar Plots + Readable Heatmap
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# -----------------------------------
# 1️⃣ Load Dataset (auto convert JSON → CSV)
# -----------------------------------
if not os.path.exists("news_category.csv"):
    print("📥 Converting JSON to CSV...")
    df = pd.read_json("C:/Users/malik/OneDrive/Desktop/Komal Dypsst/News_Category_Dataset_v3.json", lines=True)
    df = df[['category', 'headline', 'short_description']]
    df['text'] = df['headline'] + " " + df['short_description']
    df.to_csv("news_category.csv", index=False)
    print("✅ Conversion Done! Saved as news_category.csv")

df = pd.read_csv("news_category.csv")
print("\n✅ Dataset Loaded Successfully!")

# -----------------------------------
# 2️⃣ Data Cleaning
# -----------------------------------
df.drop_duplicates(inplace=True)
df['text'] = df['text'].fillna('').str.lower().str.replace('[^a-z\s]', '', regex=True)
df['category'] = df['category'].fillna('unknown')
df['text_length'] = df['text'].apply(len)

# -----------------------------------
# 3️⃣ Data Visualization
# -----------------------------------

# --- Bar Plot 1: Category Distribution (VERTICAL) ---
plt.figure(figsize=(12,6))
sns.countplot(x='category', data=df, order=df['category'].value_counts().index[:15], palette='viridis')
plt.title('📰 Top 15 Categories by Number of Articles', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Bar Plot 2: Average Text Length per Category (VERTICAL) ---
avg_length = df.groupby('category')['text_length'].mean().sort_values(ascending=False)[:15]

plt.figure(figsize=(12,6))
sns.barplot(x=avg_length.index, y=avg_length.values, palette='coolwarm')
plt.title('🧾 Average Text Length per Category (Top 15)', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Average Text Length', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Mean, Median, Mode ---
mean_len = df['text_length'].mean()
median_len = df['text_length'].median()
mode_len = df['text_length'].mode()[0]

print(f"\n📊 Mean Text Length: {mean_len:.2f}")
print(f"📊 Median Text Length: {median_len}")
print(f"📊 Mode Text Length: {mode_len}")

# -----------------------------------
# 4️⃣ Feature Extraction
# -----------------------------------
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['category']

# -----------------------------------
# 5️⃣ Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n✅ Data Split Done!")

# -----------------------------------
# 6️⃣ Train Model
# -----------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)
print("\n🤖 Model Training Complete!")

# -----------------------------------
# 7️⃣ Evaluation
# -----------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.4f}")
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# --- Confusion Matrix Heatmap (Top 10 categories only) ---
top_categories = y.value_counts().index[:10]
mask = y_test.isin(top_categories)
cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top_categories)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=top_categories, yticklabels=top_categories)
plt.title('🔥 Confusion Matrix (Top 10 Categories)', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.show()

print("\n✅ Project Completed Successfully!")
