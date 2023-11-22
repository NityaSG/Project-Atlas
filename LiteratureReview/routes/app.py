from flask import Flask, request, send_file
import requests
from xml.etree import ElementTree
import csv
import os
import openai
import io
import time
import numpy as np
import pandas as pd
import tensorflow as tf
# import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

with open('tokenizer.json') as f:
    data = f.read()
    tokenizer = tokenizer_from_json(data)

# Load the tokenizer
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

max_len = 700 
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' # out of vocabulary token
vocab_size = 3000

import openai

def generate_response(input_text):
    # Set your OpenAI API key
    openai.api_key =os.getenv('OPENAI_API_KEY')

    # Define your fixed prompt
    prompt = f"""As an AI assistant in pharmaceutical literature review, your primary task is to identify Adverse Drug Reactions (ADRs) 
in research abstracts. Focus on detecting mentions of 'adverse drug reaction', 'drug overdose', 'side effects', 
'medication error', 'death', 'drug efficacy', 'drug withdrawal', 'drug interactions', 'carcinogenicity', 'teratogenicity', 
'congenital disorder', 'drug contraindication', 'poisoning', 'substance-related disorders', 'drug resistance', 'treatment failure', 
'drug abuse', 'pregnancy complications', 'mutagenicity', and various forms of toxicity like 'neurotox', 'immunotox', 'cardiotox', 'hepatotox', 
'immunocytotox', and 'intoxication'. Analyze the abstracts for these specific terms and provide a detailed report on any identified ADRs.
Criteria:
- Identifiable patient
- Suspected adverse reaction
- Suspected medicinal product
Output: Comprehensive analysis based on the above criteria.
Note: Focus on clarity and accuracy in your assessment. {input_text}"""

    # Use OpenAI's GPT-3 model to generate a response
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      max_tokens=100
    )

    # Get the generated text
    generated_text = response.choices[0].text.strip()

    return generated_text


app = Flask(__name__)

@app.route('/fetch_articles', methods=['POST'])
def fetch_articles():
    # Get parameters from request
    query = request.form.get('query')
    date_range = request.form.get('date_range')
    num_articles = int(request.form.get('num_articles'))
    api_key = "41cacc77e4250a4e550ef9b0dbca2b168209"  # replace with your actual API key

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    db = "pubmed"

    # Fetch articles from PubMed
    search_url = f"{base_url}esearch.fcgi?db={db}&term={query}&retmax={num_articles}&sort=date&api_key={api_key}"
    search_response = requests.get(search_url)
    tree = ElementTree.fromstring(search_response.content)
    id_list = [elem.text for elem in tree.findall(".//Id")]

    # Fetch details for each article
    articles = []
    for id in id_list:
        fetch_url = f"{base_url}efetch.fcgi?db={db}&id={id}&api_key={api_key}"
        fetch_response = requests.get(fetch_url)
        article_tree = ElementTree.fromstring(fetch_response.content)

        for article in article_tree.findall(".//PubmedArticle"):
            title = article.find(".//ArticleTitle").text
            abstract = article.find(".//AbstractText").text

            articles.append({
                'name': title,
                'reference_number': id,
                'abstract': abstract,
                'full_text': 'Full text not available'
            })
        time.sleep(1)

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['name', 'reference_number', 'abstract', 'full_text'])
    writer.writeheader()
    writer.writerows(articles)

    # Convert string data to bytes
    output_bytes = io.BytesIO(output.getvalue().encode())

    # Create response
    output_bytes.seek(0)
    return send_file(output_bytes, mimetype='text/csv', as_attachment=True, attachment_filename='articles.csv')

@app.route('/',methods=['GET'])
def initialise():
    return "server is alive"

@app.route('/fetch_process', methods=['POST'])
def fetch_process():
    model = tf.keras.models.load_model('ANN')
    def predict_safety(predict_msg):
        new_seq = tokenizer.texts_to_sequences(predict_msg)
        padded = pad_sequences(new_seq,maxlen = max_len,padding = padding_type,truncating = trunc_type)
        return(model.predict(padded))
    # Get parameters from request
    query = request.form.get('query')
    date_range = request.form.get('date_range')
    num_articles = int(request.form.get('num_articles'))
    api_key = "41cacc77e4250a4e550ef9b0dbca2b168209"  # replace with your actual API key

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    db = "pubmed"

    # Fetch articles from PubMed
    search_url = f"{base_url}esearch.fcgi?db={db}&term={query}&retmax={num_articles}&sort=date&api_key={api_key}"
    search_response = requests.get(search_url)
    tree = ElementTree.fromstring(search_response.content)
    id_list = [elem.text for elem in tree.findall(".//Id")]

    # Fetch details for each article
    articles = []
    for id in id_list:
        fetch_url = f"{base_url}efetch.fcgi?db={db}&id={id}&api_key={api_key}"
        fetch_response = requests.get(fetch_url)
        article_tree = ElementTree.fromstring(fetch_response.content)

        for article in article_tree.findall(".//PubmedArticle"):
            try:
                title = article.find(".//ArticleTitle").text
                abstract = article.find(".//AbstractText").text
                authors = ', '.join([author.find(".//LastName").text + ' ' + author.find(".//ForeName").text for author in article.findall(".//Author")])
            except:
                continue
            if title is None or abstract is None or authors is None:
                continue
            try:
                prediction=predict_safety([abstract])
            except:
                prediction='NA'
            try:
                remark=generate_response([abstract])
            except:
                remark=""
            remark=generate_response([abstract])
            articles.append({
                'name': title,
                'reference_number': id,
                'authors': authors,
                'abstract': abstract,
                'prediction': prediction,
                'remark':remark
            })

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['name', 'reference_number', 'authors', 'abstract', 'prediction','remark'])
    writer.writeheader()
    writer.writerows(articles)

    # Convert string data to bytes
    output_bytes = io.BytesIO(output.getvalue().encode())

    # Create response
    output_bytes.seek(0)
    return send_file(output_bytes, mimetype='text/csv', as_attachment=True, attachment_filename='articles.csv')


if __name__ == '__main__':
    app.run(debug=True)