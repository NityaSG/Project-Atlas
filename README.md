# Primary Repository for Project ATLAS

## API Instruction : 
Endpoint: /fetch_articles

Method: POST
Description: Fetches articles from PubMed based on the provided query, date range, and number of articles. The articles are returned as a CSV file.
Parameters:
query: The search term for the articles.
date_range: The date range for the articles.
num_articles: The number of articles to fetch.
Endpoint: /fetch_process

Method: POST
Description: Fetches articles from PubMed based on the provided query, date range, and number of articles. The articles are processed using a machine learning model and returned as a CSV file.
Parameters:
query: The search term for the articles.
date_range: The date range for the articles.
num_articles: The number of articles to fetch.
Endpoint: /

Method: GET
Description: Checks if the server is alive.