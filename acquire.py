#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import os
import env
import pandas as pd
from env import host, user, password

# In[ ]:


def fetch_data(url):
    data = []

    while url:
        response = requests.get(url)
        json_data = response.json()
        data.extend(json_data["results"])
        url = json_data["next"]

    return data

def grab_csv_data(api_url, output_file):
    if not os.path.exists(output_file):
        response = requests.get(api_url)

        if response.status_code == 200:
            csv_data = response.text
            with open(output_file, 'w') as f:
                f.write(csv_data)
            print(f"CSV data saved to {output_file}")
        else:
            print(f"Request failed with status code {response.status_code}")
            return None

    return pd.read_csv(output_file)

def create_starships_dataframe():
    base_url = "https://swapi.dev/api/"
    starships_url = f"{base_url}starships/"

    starships_data = fetch_data(starships_url)
    starships_df = pd.DataFrame(starships_data)

    return starships_df

# Call the function to create the people dataframe
# starships_df = create_starships_dataframe()
# Print the first few rows of the dataframe

def create_people_dataframe():
    base_url = "https://swapi.dev/api/"
    people_url = f"{base_url}people/"

    people_data = fetch_data(people_url)
    people_df = pd.DataFrame(people_data)
    people_df.to_csv('people.csv', index=False)
    return people_df

# Call the function to create the people dataframe
#people_df = create_people_dataframe()
# Print the first few rows of the dataframe

def create_planets_dataframe():
    base_url = "https://swapi.dev/api/"
    planets_url = f"{base_url}planets/"

    planets_data = fetch_data(planets_url)
    planets_df = pd.DataFrame(planets_data)
    planets_df.to_csv('planets.csv', index=False)
    return planets_df

# Call the function to create the people dataframe
#planets_df = create_planets_dataframe()

def get_connection(db):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def acquire_superstore_data():
    filename = "superstore.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql("""SELECT categories.`Category ID`, categories.`Category`, categories.`Sub-Category`,
       customers.`Customer ID`, customers.`Customer Name`,
       departments.`department`, departments.`division`,
       orders.`Order ID`, orders.`Order Date`, orders.`Ship Date`, orders.`Ship Mode`, orders.`Customer ID`, orders.`Segment`, orders.`Country`, orders.`City`, orders.`State`, orders.`Postal Code`, orders.`Product ID`, orders.`Sales`, orders.`Quantity`, orders.`Discount`, orders.`Profit`, orders.`Category ID`, orders.`Region ID`,
       products.`Product ID`, products.`Product Name`,
       regions.`Region ID`, regions.`Region Name`
FROM orders
JOIN categories USING(`Category ID`)
JOIN customers USING(`Customer ID`)
JOIN products USING(`Product ID`)
JOIN regions USING(`Region ID`)
LEFT JOIN departments ON (departments.`division` = orders.`Segment` AND departments.`department` = categories.`Category`);""",get_connection('superstore_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        
        # Return the dataframe to the calling code
        return df 
    


def acquire_log_data():
    filename = 'anonymized-curriculum-access.txt'
    print("The acquire_log_data() function reads in data from a file named 'anonymized-curriculum-access.txt' by using the pd.read_csv() method. If the file is found in the current directory, it is read in and a Pandas DataFrame is returned. If the file is not found, the function prints 'The file doesn't exist' and recursively calls itself until the file is found.")

    if os.path.isfile(filename):
        return pd.read_csv(filename, delimiter=' ', header=None)
    else:
        print("The file doesn't exist")
        df = acquire_log_data()
        return df
    


