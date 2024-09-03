import requests
import pandas as pd
from bs4 import BeautifulSoup
import pdb
from functools import cached_property
from io import StringIO
import os
import re

class ASN_Scraper:
    def __init__(self):
        self.base_url = "https://asn.flightsafety.org"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.session = requests.Session()

        self.data_folder = "data"
        # create data folder if it doesn't exist
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    @cached_property
    
    def get_all_commercial_models(self):
        url = self.base_url + "/asndb/types/CJ"
        response = self.session.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Get the table with id myTable
        table = soup.find("table", {"id": "myTable"})

        # Lists to store data
        model_names = []
        first_flights = []
        descriptions = []
        urls = []

        # Iterate over the rows in the table body
        for row in table.find("tbody").find_all("tr"):
            # Extract the model name and URL
            model_link = row.find("td", {"class": "list"}).find("a")
            model_name = model_link.text.strip()
            model_url = self.base_url + model_link["href"]

            # Extract the first flight and description
            cells = row.find_all("td", {"class": "list"})
            first_flight = cells[1].text.strip()
            description = cells[2].text.strip()

            # Append data to lists
            model_names.append(model_name)
            first_flights.append(first_flight)
            descriptions.append(description)
            urls.append(model_url)

        # Create a DataFrame from the lists
        df = pd.DataFrame({
            "Aircraft Model": model_names,
            "First Flight": first_flights,
            "Description": descriptions,
            "URL": urls
        })

        return df

    def _get_num_pages(self, page_url):

        # if page_url is a full url, use it as is
        if page_url.startswith(self.base_url):
            url = page_url
        else:
            url = self.base_url + page_url
        response = self.session.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # find div with class pagenumbers
        page_numbers = soup.find("div", {"class": "pagenumbers"})

        if page_numbers is None:
            return 1
        
        # count number of 'a' tags within the div
        num_pages = len(page_numbers.find_all("a"))

        return int(num_pages) + 1 # add 1 to account for the current page which is a span tag

    def get_model_data(self, model_url):
        num_pages = self._get_num_pages(model_url)

        all_pages_df = []

        for page in range(1, num_pages + 1):
            # Loop through all pages for the model
            url = model_url + f"/{page}"
            response = self.session.get(url, headers=self.headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                table = soup.find("table", {"class": "hp"})

                if table:
                    # Convert the HTML table into a pandas DataFrame
                    df = pd.read_html(StringIO(str(table)))[0]
                    all_pages_df.append(df)
                else:
                    print(f"No table found on page {page}")
            else:
                print(f"Failed to retrieve page {page}, status code: {response.status_code}")

        if all_pages_df:
            # Concatenate all DataFrames into one
            final_df = pd.concat(all_pages_df, ignore_index=True)

            # clean, remove unnamed columns
            final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]

            return final_df
        else:
            print("No data found")
            return None
        

    def sanitize_model_name(self, name):
        # Replace problematic characters with safe ones
        sanitized_name = re.sub(r'[<>:"/\\|?*]', '-', name)
        sanitized_name = sanitized_name.strip()  # Remove leading/trailing whitespace
        return sanitized_name

    def save_all_models_data(self):
        commercial_models = self.get_all_commercial_models

        # Loop through each row in the DataFrame
        for index, row in commercial_models.iterrows():
            model = row["Aircraft Model"]
            description = row["Description"]
            
            # Replace spaces in the model name with hyphens
            model = model.replace(" ", "-")

            # Append description to the model name if it exists
            if description:
                model = f"{model} - {description.replace(' ', '-')}"

            # Sanitize the model name to ensure it's a valid filename
            model = self.sanitize_model_name(model)

            # Define the file path
            file_path = f"{self.data_folder}/{flight_models}/{model}.csv"

            # Check if the file already exists
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping.")
                continue

            # Get the data for the model
            model_url = row["URL"]
            data = self.get_model_data(model_url)

            # Save the data to CSV if it's not None
            if data is not None:
                data.to_csv(file_path, index=False)
                print(f"Data for {model} saved to {file_path}")
            else:
                print(f"No data found for {model}")

if __name__ == "__main__":
    scraper = ASN_Scraper()
    # models = scraper.get_all_commercial_models()

    scraper.get_all_commercial_models


    # num_pages = scraper._get_num_pages("/asndb/type/A320")

    # data = scraper.get_model_data("Airbus A320")
    scraper.save_all_models_data()
    pdb.set_trace()