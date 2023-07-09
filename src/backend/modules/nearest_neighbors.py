import pandas as pd
import pickle
import re

from app import app
from flask import request
from flask import jsonify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from selenium import webdriver
from selenium.webdriver.common.by import By


@app.route('/nearest_neighbors', methods=['GET'])
def nearest_neighbors():
    """Scrape amazon data and return similar ebay results w/ NearestNeighbors model"""

    amazon_url = request.args.get('url')
    # Generate user agent
    op = webdriver.ChromeOptions()
    op.add_argument('headless')
    driver = webdriver.Chrome(options=op)
    driver.get(amazon_url)
    title_string = driver.find_element(By.ID, 'productTitle').text

    driver.close()

    with open('../models/count_vec.pkl', 'rb') as f:
        (count_vec, title_tfidf, clf) = pickle.load(f)

    # Text feature extraction
    sw = set(stopwords.words('english'))
    # Remove special characters
    title_string_clean = re.sub(r'\s+', ' ', re.sub('[^A-Za-z0-9]', ' ', title_string.strip().lower())).strip()
    # Tokenize
    title_string_clean = word_tokenize(title_string_clean)
    # Remove stop words
    title_string_clean = [w for w in title_string_clean if w not in sw]
    title_string_clean = " ".join(title_string_clean)

    # Get nearest neighbors
    title_vec = count_vec.transform([title_string_clean])
    norm_title_vec = title_tfidf.transform(title_vec)
    nbrs = clf.kneighbors(norm_title_vec, 10, return_distance=False)

    df = pd.read_csv('../data/ebay_data_clean.csv')

    urls = df.iloc[nbrs[0]]['Pageurl']

    return jsonify(urls.tolist())