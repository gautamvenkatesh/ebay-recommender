# ebay-recommender

#### Description
This application uses NLP techniques and NearestNeighbors to find ebay listings for given amazon listings. For example, if I had a amazon link for headphones, querying this API would return a list of ebay links with similar products. UI coming soon!

#### Technology
This project uses the NLTK library to tokenize and clean textual data. It then uses sklearn's implementation of traditional NearestNeighbors with cosine distance as a metric. Alternatively, the program can be run as ball tree for better performance. Backend API is written with Flask framework.

<img width="1454" alt="image" src="https://github.com/gautamvenkatesh/ebay-recommender/assets/48636589/c568260f-5024-4132-b3cf-a9213a2a4a49">
