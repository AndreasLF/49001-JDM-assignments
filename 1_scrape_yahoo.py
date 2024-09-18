import requests
from bs4 import BeautifulSoup
import pandas as pd
import pdb

# The URL of the Yahoo Finance page
# url = 'https://finance.yahoo.com/quote/AIR.PA/history/?period1=999500400&period2=1726541609'  # Airbus
url = 'https://finance.yahoo.com/quote/BA/history/?period1=-252322200&period2=1726542120' # Boeing


# extract after quote/ and before /history
stock = url.split('quote/')[1].split('/history')[0]

headers = {'User-Agent': 'Mozilla/5.0'}

# Send an HTTP request to the page and get the content
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table
table = soup.find('table', {'class': 'table'})

# Create lists to hold the table data
dates = []
opens = []
highs = []
lows = []
closes = []
adj_closes = []
volumes = []

# Loop over the rows in the table body
for row in table.find('tbody').find_all('tr'):
    cols = row.find_all('td')
    if len(cols) == 7:
        dates.append(cols[0].text.strip())
        opens.append(cols[1].text.strip())
        highs.append(cols[2].text.strip())
        lows.append(cols[3].text.strip())
        closes.append(cols[4].text.strip())
        adj_closes.append(cols[5].text.strip())
        volumes.append(cols[6].text.strip())

# Create a DataFrame with the scraped data
df = pd.DataFrame({
    'Date': dates,
    'Open': opens,
    'High': highs,
    'Low': lows,
    'Close': closes,
    'Adj Close': adj_closes,
    'Volume': volumes
})

# Save the DataFrame to a CSV file
df.to_csv(f'data/yahoo_finance_data_{stock}.csv', index=False)

print(f'Data saved to data/yahoo_finance_data_{stock}.csv')
