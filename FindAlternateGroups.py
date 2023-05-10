import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK objects
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess tags
def preprocess_tags(tags):
    # Join all tags into a single string
    tags = ' '.join(tags)
    # Tokenize the string
    tokens = nltk.word_tokenize(tags.lower())
    # Filter out non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]
    # Lemmatize the tokens and filter out stop words
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # Join the tokens back into a string
    strs = ' '.join(tokens)
    return strs

# Function to find alternate groups
def FindAlternateGroups(domain, threshold=0.50):
    # Construct URL and API endpoint
    url = f'https://{domain}/collections/all'
    endpoint = '/products.json'

    # API request URL
    api_url = url + endpoint

    # Make a GET request to the API
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data
        json_data = json.loads(response.text)
        # Create a DataFrame from the JSON data
        df = pd.json_normalize(json_data['products'])
        products_df = pd.DataFrame({
                'title':df['title'],
                'handle': df['handle'],
                'product ID': df['id'],
                'vendor': df['vendor'],
                'tags' : df['tags'],
                'product Type': df['product_type'],
                'price': df['variants'].apply(lambda x: x[0]['price']),
                'sKU': df['variants'].apply(lambda x: x[0]['sku']),

            })
    # Create a copy of the dataframe with only the relevant columns
    product_df = df[['title', 'handle', 'tags']].copy()

    # Preprocess the product tags
    product_df['tags'] = product_df['tags'].apply(preprocess_tags)

    # Compute the cosine similarity matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_df['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Create an empty dictionary to store the alternate groups
    alternate_groups = {}

    # Loop through each product in the dataframe
    for idx, product in product_df.iterrows():
        # Get the cosine similarity scores of all products
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Filter products based on similarity threshold
        sim_scores = [score for score in sim_scores if score[1] >= threshold]
        # Get the product indices
        product_indices = [i[0] for i in sim_scores]
        # Exclude the current product from the indices
        product_indices = [i for i in product_indices if i != idx]
        # If there are alternate products, add them to the dictionary
        if product_indices:
            handles = product_df.loc[product_indices]['handle'].tolist()
            urls = [f'https://{domain}/products/{h}' for h in handles]
            alternate_groups[product['title']] = urls

    # Return the alternate groups as a JSON
    return json.dumps(alternate_groups)

if __name__ == '__main__':
    domain = input("Enter Store Domain (e.g. www.boysnextdoor-apparel.co):     ")
    
    if domain.startswith('http'):
        domain = domain.split('/')[2]
        print(domain)
        ans = FindAlternateGroups(domain)
        with open(f'alternate_groups {domain}.json', 'w') as f:
          json.dump(ans, f)
    elif domain.startswith('www'):
        domain = domain.split('/')[0]
        print(domain)
        ans = FindAlternateGroups(domain)
        with open(f'alternate_groups {domain}.json', 'w') as f:
          json.dump(ans, f)
    else: print("enter correct domain")