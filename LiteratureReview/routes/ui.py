import streamlit as st
import requests
import pandas as pd
import io
from datetime import datetime, timedelta

# Define the URL of the API endpoint
api_url = "https://atlas-9seq.onrender.com/fetch_process"
st.title('Project ATLAS')
st.caption("ThinkRoman Ventures")
#st.warning("Only for testing purpose. Not in production")
# Create a form for user input
with st.form(key='my_form'):
    query = st.text_input(label='Enter your query')
    date_range = st.date_input('Select a date range', value=[datetime.today() - timedelta(days=7), datetime.today()])
    num_articles = st.number_input('Enter the number of articles', min_value=1, value=10, step=1)
    submit_button = st.form_submit_button(label='Submit')

# When the user clicks the "Submit" button, send a POST request to the API
if submit_button:
    response = requests.post(
        api_url,
        data={
            'query': query,
            'date_range': ','.join([date.strftime("%Y-%m-%d") for date in date_range]),
            'num_articles': num_articles
        }
    )

    # If the request was successful, display the returned CSV data
    if response.status_code == 200:
        data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        st.write(data)
    else:
        st.write("Error: API request failed")
