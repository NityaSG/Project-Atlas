import streamlit as st
import requests
import pandas as pd
import io
from datetime import datetime, timedelta

# Define the URLs of the API endpoints
fetch_api_url = "http://ec2-54-211-188-5.compute-1.amazonaws.com/"
upload_api_url = "https://atlas-web-thjf.onrender.com/upload_csv"

st.title('Project ATLAS')
st.caption("ThinkRoman Ventures")
# st.warning("Only for testing purpose. Not in production")

option = st.radio("Choose an option:", ('Use API to fetch data', 'Upload a CSV file'))

with st.form(key='my_form'):
    if option == 'Use API to fetch data':
        query = st.text_input(label='Enter your query')
        date_range = st.date_input('Select a date range', value=[datetime.today() - timedelta(days=7), datetime.today()])
        num_articles = st.number_input('Enter the number of articles', min_value=1, value=10, step=1)
        submit_button = st.form_submit_button(label='Fetch Data')
    elif option == 'Upload a CSV file':
        st.markdown("""
        **CSV File Format Instructions:**
        - Ensure your CSV file has the following headers: `title`, `abstract`, `full_text`.
        - `title`: Title of the article.
        - `abstract`: Abstract of the article.
        - `full_text`: Full text of the article.
        """)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        submit_button = st.form_submit_button(label='Upload')

if submit_button:
    if option == 'Use API to fetch data':
        # Send a POST request to the fetch API
        response = requests.post(
            fetch_api_url,
            data={
                'query': query,
                'date_range': ','.join([date.strftime("%Y-%m-%d") for date in date_range]),
                'num_articles': num_articles
            }
        )
        if response.status_code == 200:
            data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            st.write(data)
        else:
            st.write("Error: API request failed")
    elif option == 'Upload a CSV file':
        # Send the CSV file to the upload API
        if uploaded_file is not None:
            files = {'file': uploaded_file.getvalue()}
            response = requests.post(upload_api_url, files=files)
            if response.status_code == 200:
                # Assuming the response contains the processed data
                data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                st.write(data)
            else:
                st.write("Error: File upload failed")
        else:
            st.write("Please upload a CSV file.")
