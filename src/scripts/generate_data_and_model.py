import numpy as np
import nltk
import pandas as pd
import pickle
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors

nltk.download('stopwords')
nltk.download('punkt')


def generate_data_and_model():
    """Clean dataset, apply feature engineering, save clean data & model"""

    names = ["Uniq Id", "Crawl Timestamp", "Pageurl", "Website", "Title", "Num Of Reviews", "Average Rating", "Number Of Ratings", "Model Num", "Sku", "Upc", "Manufacturer", "Model Name", "Price", "Monthly Price", "Stock", "Carrier", "Color Category"," Internal Memory", "Screen Size", "Specifications", "Five Star", "Four Star", "Three Star", "Two Star", "One Star", "Discontinued", "Broken Link", "Seller Rating", "Seller Num Of Reviews", "extra"]
    df_raw = pd.read_csv("../data/marketing_sample_for_ebay_com-ebay_com_product__20210101_20210331__30k_data.csv", names=names, skiprows=[0])
    
    df = df_raw[['Uniq Id', 'Pageurl', 'Title', 'Num Of Reviews', 'Average Rating', 'Price', 'Specifications']]

    # Cleaning
    df = df.dropna(subset=['Title', 'Price'])
    df['Price'] = df['Price'].apply(lambda p: float(p.replace('$', '').replace(',', '')))

    # Text feature extraction
    sw = set(stopwords.words('english'))
    # Remove special characters
    df['title_tok'] = df['Title'].apply(lambda title: re.sub(r'\s+', ' ', re.sub('[^A-Za-z0-9]', ' ', 
            title.strip().lower())).strip())
    # Tokenize
    df['title_tok'] = df['title_tok'].apply(word_tokenize)
    # Remove stop words
    df['title_tok'] = df['title_tok'].apply(lambda word_l: [w for w in word_l if w not in sw])
    df['title_tok'] = df['title_tok'].apply(lambda word_l: " ".join(word_l))

    # Bag of words implement
    count_vec = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    nn = NearestNeighbors(algorithm='brute', metric="cosine")

    count_matrix = count_vec.fit_transform(df['title_tok'])
    title_tfidf = tfidf_transformer.fit_transform(count_matrix)
    clf = nn.fit(title_tfidf)

    df['title_tok_vec'] = title_tfidf.getnnz()

    # Save data and model
    with open('../models/count_vec.pkl', 'wb') as fout:
        pickle.dump((count_vec, tfidf_transformer, clf), fout)
    df.to_csv("../data/ebay_data_clean.csv")


def main():
    generate_data_and_model()


if __name__ == "__main__":
    main()