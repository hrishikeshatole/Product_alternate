import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests



def create_df(domain):
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
        return products_df

def FindAlternateGroups(product_title, domain, threshold=0.50):
    df = create_df(domain)
    tags = df['tags'].apply(lambda x: ' '.join(x))
    vectorizer = TfidfVectorizer()
    tag_vectors = vectorizer.fit_transform(tags)
    cosine_sim = cosine_similarity(tag_vectors)
    # Get the index of the product
    idx = df[df['title'] == product_title].index[0]
    # Get the cosine similarity scores of all products
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Filter products based on similarity threshold
    sim_scores = [score for score in sim_scores if score[1] >= threshold]
    # Get the product indices
    product_indices = [i[0] for i in sim_scores]
    
    # Initialize a list to hold the URLs
    urls = []

    # Loop through the product indices and get the handles
    alternates = []
    for index in product_indices:
        handle = df.loc[index]['handle']
        url = f"https://{domain}/products/{handle}"
        alternates.append(url)

    # Return the alternate products as a JSON
    output = {"product alternates": alternates}
    return json.dumps(output)

product_title = "Ben Davis Heavy Duty Pocket Tee Ash Grey"
domain = "www.boysnextdoor-apparel.co"
recommendations = FindAlternateGroups(product_title, domain)
print(recommendations)
with open(f'alternate_groups {product_title}.json', 'w') as f:
          json.dump(recommendations, f)

