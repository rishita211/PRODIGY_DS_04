import zipfile
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcpyloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Define zip path and extract
zip_path = r"C:\Users\RISHITA\Downloads\twitter_training.csv.zip"
extract_path = r"C:\Users\RISHITA\Downloads\twitter_dataset"

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Find the CSV file
csv_file = None
for file in os.listdir(extract_path):
    if file.endswith(".csv"):
        csv_file = os.path.join(extract_path, file)
        break

# Check if CSV found
if csv_file is None:
    raise FileNotFoundError("CSV file not found in the ZIP!")

# Load CSV
df = pd.read_csv(csv_file, encoding='latin1', header=None)
df.columns = ['ID', 'Sentiment', 'Entity', 'Tweet']

# Display basic info
print(df.head())

# Sentiment Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Sentiment', data=df, palette='Set2')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.show()

# Word Cloud for Positive Sentiment
positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['Tweet'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white',
                      stopwords=set(stopwords.words('english'))).generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Positive Tweets")
plt.show()
