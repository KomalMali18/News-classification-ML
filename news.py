# ==========================================
# 📰 News Article Categorization (CSV)
# Multinomial Naive Bayes + Vertical Bar Plots
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------
# 1. Load Dataset (CSV)
# ---------------------------
df = pd.read_csv(r'c:/Users/PC/AppData/Local/Temp/Rar$DRa4812.25306/news-article-categories.csv')

# Print columns to verify
print("Columns in CSV:", df.columns)

# ---------------------------
# 2. Select relevant columns
# ---------------------------
text_col = 'title'        # use 'title' column as text
category_col = 'category' # category column

df = df[[text_col, category_col]].copy()
df.rename(columns={text_col:'text', category_col:'category'}, inplace=True)

# ---------------------------
# 3. Data Cleaning
# ---------------------------
df = df.drop_duplicates()
df['text'] = df['text'].fillna('').str.lower()
df['category'] = df['category'].fillna('unknown')

# Remove punctuation
df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)

# Calculate text length
df['text_length'] = df['text'].apply(len)

# ---------------------------
# 4. Data Visualization (Vertical Bar Plots)
# ---------------------------

# Average text length per category
avg_text_length = df.groupby('category')['text_length'].mean().sort_values(ascending=False)

plt.figure(figsize=(14,6))
sns.barplot(x=avg_text_length.index, y=avg_text_length.values, palette='viridis')
plt.xlabel('Category')
plt.ylabel('Average Text Length')
plt.title('Average Text Length by News Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Number of articles per category
plt.figure(figsize=(14,6))
sns.countplot(x='category', data=df, order=df['category'].value_counts().index, palette='magma')
plt.xlabel('Category')
plt.ylabel('Number of Articles')
plt.title('Number of Articles per Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print Mean, Median, Mode
print(f"\nMean Text Length: {df['text_length'].mean():.2f}")
print(f"Median Text Length: {df['text_length'].median()}")
print(f"Mode Text Length: {df['text_length'].mode()[0]}")

# ---------------------------
# 5. Feature Extraction
# ---------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['category']

# ---------------------------
# 6. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 7. Train Multinomial Naive Bayes
# ---------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------------------
# 8. Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(12,8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.show()
