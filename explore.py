#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import env
from env import host, user, password
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import wrangle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# In[ ]:

import pandas as pd

def data_dict_anomaly():
    data_dict = {
        'page': '/',
        'user_id': 1,
        'cohort_id': 8,
        'ip': '97.105.19.61',
        'datetime': '2018-01-26 09:55:03',
        'time_spent': 0.00,
        'session_id': 0,
        'location': 'Texas, United States',
        'category': 'curriculum_infra',
        'cohort': 'Crab',
        'time_spent_zscore': -0.14,
        'avg_time_spent_by_cohort': 454.60
    }

    df = pd.DataFrame([data_dict])
    print("The data dictionary contains information about a single user session on a website. It includes the page accessed, the user ID, the cohort ID, the IP address of the user, the datetime of the session, the amount of time spent on the page, the session ID, the location of the user, the category of the content, the cohort name, the z-score of the time spent on the page, and the average time spent by the cohort on that page. This information can be used to analyze user behavior and engagement with the website.")
    return df


import matplotlib.pyplot as plt

def visualize_visits_by_country(df):
    # Use ~ to negate the condition and select non-US locations
    non_us_locations = df[~df['location'].str.contains('United States', na=False)]

    non_us_locations['country'] = non_us_locations['location'].str.split(',').str[0]
    # Group by country and count the number of visits
    visits_by_country = non_us_locations.groupby('country')['page'].count().reset_index()

    # Filter countries that visited the page only once
    visits_by_country = visits_by_country[visits_by_country['page'] > 5]

    # Sort the values in descending order
    visits_by_country = visits_by_country.sort_values('page', ascending=False)

    # Create the horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(visits_by_country['country'], visits_by_country['page'])
    ax.set_xlabel('Number of Page Visits')
    ax.set_ylabel('Country')
    ax.set_title('Page Visits by Country (Excluding US)')
    print("Visualize Suspicious Activity By Country: This function generates a horizontal bar chart that shows the number of page visits by country, excluding the United States. The purpose of this visualization is to increase understanding of the web traffic distribution of web traffic at CodeUp. The visualization filters out countries that visited the page only once and sorts the remaining countries by the number of page visits in descending order. This activity should be throughly investigted, but be warned that many anonymous/private people use Virtual Private Networks (VPN) to surf the interwebs.")
    plt.show()

import matplotlib.pyplot as plt

def favorite_learning_page(df):
    # Top 10 pages for web_development
    webdev_top10 = pd.DataFrame({
        'page': ['index.html', 'jquery/ajax/weather-map', 'java-ii', 'java-iii', 'javascript-i',
                 'javascript-i', 'javascript-ii', 'java-iii', 'java-ii', 'html-css'],
        'visits': [1477, 811, 454, 452, 388, 380, 375, 314, 301, 280]
    })

    # Create the horizontal bar chart for web development
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(webdev_top10['page'], webdev_top10['visits'])
    ax.set_xlabel('Number of Visits')
    ax.set_ylabel('Page')
    ax.set_title('Top 10 Pages for Web Development')

    # Show the visualization and print a description
    plt.show()
    print("This visualization shows the top 10 pages visited by graduates in the Web Development program, based on the number of visits.")

    # Top 10 pages for data_science
    datasci_top10 = pd.DataFrame({
        'page': ['sql/mysql-overview', 'classification/overview', 'classification/scale_features_or_not.svg',
                 '6-regression/1-overview', '10-anomaly-detection/1-overview', '10-anomaly-detection/AnomalyDetectionCartoon.jpeg',
                 'mysql', 'appendix/cli-git-overview', 'classification/overview', '3-sql/1-mysql-overview'],
        'visits': [487, 466, 356, 342, 332, 332, 282, 247, 241, 231]
    })

    # Create the horizontal bar chart for data science
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(datasci_top10['page'], datasci_top10['visits'])
    ax.set_xlabel('Number of Visits')
    ax.set_ylabel('Page')
    ax.set_title('Top 10 Pages for Data Science')

    # Show the visualization and print a description
    plt.show()
    print("This visualization shows the top 10 pages visited by graduates in the Data Science program, based on the number of visits.")

    
import pandas as pd
import matplotlib.pyplot as plt

def suspect_IP(df):
    # Select rows with a specific IP address
    access_log = df[df['ip'].str.contains('97.105.19.58')]
    
    # Convert 'datetime' column to datetime format and set it as the index
    access_log['datetime'] = pd.to_datetime(access_log['datetime'])
    access_log = access_log.set_index('datetime')
    
    # Resample by day and count the number of visits
    visits_per_day = access_log.resample('D')['page'].count()
    
    # Create the line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(visits_per_day.index, visits_per_day.values)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Visits')
    ax.set_title('Number of Visits by Day for Suspect IP Address')
    plt.show()
    print("This function visualizes the number of visits per day for a specific IP address (97.105.19.58), which is suspected of engaging in suspicious activity on the website.")    
    
    

def avg_team_ot(df):
    # Calculate the average overtime across all teams
    avg_overtime = df['over_time'].mean()

    # Find the three teams with the most overtime
    top_3_overtime_teams = df.groupby('team')['over_time'].mean().sort_values(ascending=False).head(3).index

    # Create a new column to indicate if the team is in the top 3
    df['highlight'] = df['team'].apply(lambda x: 'Top 3' if x in top_3_overtime_teams else 'Others')

    plt.figure(figsize=(10, 6))

    # Plot the bar chart, highlighting the top 3 teams
    sns.barplot(x='team', y='over_time', data=df, hue='highlight', palette={'Top 3': 'gray', 'Others': 'blue'}, dodge=False)

    # Add the average overtime h-line
    plt.axhline(avg_overtime, color='green', linestyle='--', label='Average Overtime')

    plt.title('Overtime by Team')
    plt.xlabel('Team')
    plt.ylabel('Overtime (minutes)')
    plt.legend(title='Team Category', loc='upper right')

    plt.show()

def incentive_over_time(df):
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Calculate the interquartile range (IQR) to detect and remove outliers
    Q1 = df['incentive'].quantile(0.25)
    Q3 = df['incentive'].quantile(0.75)
    IQR = Q3 - Q1

    # Filter out the outliers
    filtered_df = df[(df['incentive'] >= (Q1 - 1.5 * IQR)) & (df['incentive'] <= (Q3 + 1.5 * IQR))]

    # Group the data by week and calculate the average incentive per week
    weekly_incentive = filtered_df.groupby(pd.Grouper(key='date', freq='W'))['incentive'].mean().reset_index()

    # Plot the line chart
    plt.plot(weekly_incentive['date'], weekly_incentive['incentive'])
    plt.xlabel('Date')
    plt.ylabel('Average Incentive')
    plt.title('Incentive Over Time (Per Week)')
    plt.xticks(rotation=45)
    plt.show()
    
    
def plot_production_data(df):
    """
    Plot a bar chart of prod_capacity and actual_production by team, and add a horizontal line
    for the average actual_production.
    Args:
        df: a pandas DataFrame containing production data with columns 'team', 'prod_capacity', and 'actual_production'
    Returns:
        None
    """
    # Calculate the max actual_production
    max_actual_production = df['actual_production'].mean()

    # Group the data by team and calculate the mean of actual_productivity, prod_capacity, and actual_production
    grouped_data = df.groupby('team')[['prod_capacity', 'actual_production']].mean()

    # Plot the bar chart
    grouped_data.plot(kind='bar', figsize=(10, 6))

    # Add a horizontal line for the max actual_production
    plt.axhline(max_actual_production, color='red', linestyle='--', label=f'Average Actual Production: {max_actual_production:.2f}')

    # Set the title and axis labels
    plt.title('Prod Capacity, and Actual Production by Team')
    plt.xlabel('Team')
    plt.ylabel('Value')
    plt.legend()

    # Show the plot
    plt.show()
    
    
def plot_monthly_productivity_by_team(df):
    # Convert the 'date' column to datetime type if not already
    df['date'] = pd.to_datetime(df['date'])

    # Extract the month from the 'date' column
    df['month'] = df['date'].dt.to_period('M')

    # Calculate the average actual_productivity across all teams
    avg_productivity = df['actual_productivity'].mean()

    # Factory productivity standard
    factory_standard = 0.75

    # Get the list of unique months and teams
    unique_months = df['month'].unique()
    unique_teams = df['team'].unique()

    # Create subplots for each month
    n_months = len(unique_months)
    fig, axes = plt.subplots(n_months, 1, figsize=(10, 5 * n_months), sharex=True)

    for i, month in enumerate(unique_months):
        # Filter data for the current month
        month_data = df[df['month'] == month]

        # Calculate the mean actual_productivity for each team in the current month
        team_productivity = month_data.groupby('team')['actual_productivity'].mean()

        # Find the team with the highest actual_productivity
        max_team = team_productivity.idxmax()

        # Set bar colors
        colors = ['tab:gray' if team != max_team else 'tab:orange' for team in unique_teams]

        # Create a bar chart
        axes[i].bar(unique_teams, team_productivity, color=colors)

        # Add horizontal lines
        axes[i].axhline(avg_productivity, color='green', linestyle='--', label='Average Productivity')
        axes[i].axhline(factory_standard, color='red', linestyle='--', label='Factory Standard')

        # Set labels and title
        axes[i].set_xlabel('Team')
        axes[i].set_ylabel('Actual Productivity')
        axes[i].set_title(f'Actual Productivity by Team for {month}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()
    
    
def plot_incentive_per_team(df):
    # Calculate the average incentive per team
    grouped_df = df.groupby('team')['incentive'].mean().reset_index()

    # Calculate the overall average incentive across all teams
    average_incentive = df['incentive'].mean()

    # Find the top 3 teams by their average incentives
    top_3_teams = grouped_df.nlargest(3, 'incentive')['team']

    # Create a custom color list, setting 'red' for top 3 teams and 'blue' for others
    colors = ['grey' if team in top_3_teams.values else 'blue' for team in grouped_df['team']]

    # Plot the bar chart
    plt.bar(grouped_df['team'].astype(str), grouped_df['incentive'], color=colors)
    plt.axhline(average_incentive, color='g', linestyle='--', label='Average Incentive')

    plt.xlabel('Team')
    plt.ylabel('Incentive')
    plt.title('Average Incentive per Team')
    plt.legend()
    plt.show()
    
    
def plot_actual_productivity(df):
    """
    Plots a bar chart of the actual productivity by team, highlighting the team with the highest productivity in blue.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the actual productivity data by team.

    Returns:
    None.
    """

    # Calculate the average actual_productivity across all teams
    average_productivity = df['actual_productivity'].mean()

    # Group the data by team and calculate the average actual_productivity for each team
    grouped_df = df.groupby('team')['actual_productivity'].mean().reset_index()

    # Find the team with the highest actual_productivity
    highest_team = grouped_df.loc[grouped_df['actual_productivity'].idxmax()]['team']

    # Plot the bar chart with team numbers on x-axis and actual_productivity on y-axis
    # Highlight the highest team in blue and others in light blue
    bar_colors = ['blue' if team == highest_team else 'lightblue' for team in grouped_df['team']]
    plt.bar(grouped_df['team'], grouped_df['actual_productivity'], color=bar_colors)

    # Add a horizontal red dashed line representing the average actual_productivity
    plt.axhline(average_productivity, color='red', linestyle='--', label=f'Average: {average_productivity:.2f}')

    # Label the axes and add a title
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')
    plt.title('Actual Productivity by Team')
    plt.legend()
    plt.show()
    
    
import matplotlib.pyplot as plt

def plot_top_pages(df):
    # View only the web_development category
    web_dev_df = df[df['category'] == 'web_development']

    # Group the data by page, and count the number of pageviews
    grouped_pages = web_dev_df.groupby('page').size().reset_index(name='pageviews')

    # Sort the values by pageviews in descending order, and select the top 5 pages
    most_active_pages = grouped_pages.sort_values('pageviews', ascending=False).head(5)

    # Create a horizontal bar chart to show the number of pageviews for each page
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(most_active_pages['page'], most_active_pages['pageviews'])
    ax.set_xlabel('Pageviews')
    ax.set_ylabel('Page')
    ax.set_title('Top 5 Pages by Pageviews')

    # Save the figure
    plt.savefig('top_5_pages.png', bbox_inches='tight')

    # Print out a summary
    print('Top 5: Web Development Lesson Views')
    print('This horizontal bar chart shows the top 5 pages that attract the most traffic consistently. The most popular page appears to be "javascript-i", "java-iii", and "html-css". There is a significant drop in traffic between the top page and the rest of the top 5. It is important to note that this analysis only considers the pages in the dataset, and may not reflect the entire website traffic.')
    
    
def ds_top_pages(df):
    ds_df = df[df['category'] == 'data_science']
    
    # Group the data by page, and count the number of pageviews
    grouped_pages = ds_df.groupby('page').size().reset_index(name='pageviews')

    # Sort the values by pageviews in descending order, and select the top 5 pages
    most_active_pages = grouped_pages.sort_values('pageviews', ascending=False).head(5)

    # Create a bar chart to show the number of pageviews for each page
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(most_active_pages['page'], most_active_pages['pageviews'])
    ax.set_xlabel('Pageviews')
    ax.set_ylabel('Page')
    ax.set_title('Top 5 Pages by Pageviews')

    # Save the figure
    plt.savefig('top_5_pages.png', bbox_inches='tight')

    # Print out a summary
    print('Top 5: Data Science Lesson Views')
    print('This bar chart shows the top 5 pages that attract the most traffic consistently. The most popular page appears to be "mysql", followed by "mysql/tables", and "mysqldatabase". There is a significant drop in traffic between the top page and the rest of the top 5. It is important to note that this analysis only considers the pages in the dataset, and may not reflect the entire website traffic.')
    
import pandas as pd
import random

def explore_prep(df):
    # Assign random fish names to each cohort_id based on category
    web_dev_animal_names = ['Lion', 'Tiger', 'Leopard', 'Cheetah', 'Jaguar', 'Panther', 'Cougar', 'Lynx', 'Bobcat', 'Ocelot', 'Caracal', 'Serval', 'Wolf', 'Fox', 'Coyote', 'Jackal', 'Raccoon', 'Skunk', 'Badger', 'Weasel', 'Ferret', 'Mink', 'Otter', 'Hyena', 'Honey Badger', 'Puma', 'Wolverine', 'Gazelle', 'Antelope', 'Elk', 'Moose', 'Reindeer', 'Giraffe', 'Zebra', 'Hippopotamus', 'Rhino', 'Elephant', 'Gorilla', 'Chimpanzee', 'Orangutan', 'Baboon', 'Marmoset', 'Lemur', 'Sloth', 'Kangaroo', 'Koala', 'Wombat', 'Platypus', 'Emu', 'Ostrich']
    data_sci_fish_names = ['Salmon', 'Trout', 'Bass', 'Mullet', 'Carp', 'Goby', 'Pike', 'Tilapia', 'Sturgeon', 'Sardine', 'Swordfish', 'Anchovy', 'Herring', 'Whiting', 'Ling', 'Pilchard', 'Hake', 'John Dory', 'Red Snapper', 'Sea Bass', 'Dolphin Fish', 'Gurnard', 'Turbot', 'Octopus', 'Squid', 'Cuttlefish', 'Crab', 'Lobster', 'Prawn', 'Shrimp', 'Clam', 'Oyster', 'Mussel', 'Cockle', 'Scallop', 'Abalone', 'Periwinkle', 'Barnacle', 'Jellyfish', 'Sea Cucumber', 'Sea Urchin', 'Starfish', 'Seahorse', 'Nautilus', 'Sponge', 'Corals', 'Anemone', 'Sea Slug']
    cohort_names = {}
    for i, row in df.iterrows():
        if row['category'] == 'web_development':
            nick_name = random.choice(web_dev_animal_names)
        else:
            nick_name = random.choice(data_sci_fish_names)
        if row['cohort_id'] not in cohort_names:
            cohort_names[row['cohort_id']] = nick_name
        df.loc[i, 'nick_name'] = nick_name
        
    # Merge the new df back into the original df
    df = pd.merge(df, pd.DataFrame(cohort_names.items(), columns=['cohort_id', 'nick_name']), on='cohort_id')

    # Drop 'fish_name_x' column and rename 'fish_name_y' to 'cohort'
    df = df.drop('nick_name_y', axis=1)
    df = df.rename(columns={'nick_name_x': 'cohort'})

    df['time_spent_zscore'] = (df['time_spent'] - df['time_spent'].mean()) / df['time_spent'].std()

    df['avg_time_spent_by_cohort'] = df.groupby(['cohort', 'page'])['time_spent'].transform('mean')

    df.to_csv('prelim_clean.csv', index=False)
    print("Assigns random names to each cohort based on the category of that cohort. There are two lists of names: one for web development cohorts and the other for data science cohorts. For each row of the dataframe, a random name is chosen from the appropriate list based on the category of the cohort. This name is added to a dictionary cohort_names where the keys are cohort IDs and the values are the corresponding nicknames. The dataframe is merged with a new dataframe created from the cohort_names dictionary. The cohort_id column is used as the key for the merge. The 'nick_name_y' column is dropped and the 'nick_name_x' column is renamed to 'cohort'. A new column 'time_spent_zscore' is created which contains the z-score of the time spent by each row. A new column 'avg_time_spent_by_cohort' is created which contains the mean time spent on a page by a particular cohort. The cleaned dataframe is saved to a CSV file named 'prelim_clean.csv'.")
    return df


def analyze_cohort(df):
    # Calculate time spent z-score and average time spent by cohort and page
    df['time_spent_zscore'] = (df['time_spent'] - df['time_spent'].mean()) / df['time_spent'].std()
    df['avg_time_spent_by_cohort'] = df.groupby(['cohort', 'page'])['time_spent'].transform('mean')

    # List of pages to omit
    omit_pages = ['/', 'search/search_index.json', 'toc', 'spring', 'appendix']

    # Group by 'cohort_id' and 'page' and calculate average time spent
    cohort_page_time = df[~df['page'].isin(omit_pages)].groupby(['cohort_id', 'page'])['time_spent'].mean().reset_index()

    # Find the cohort and page combination with the highest average time spent
    max_cohort_page_time = cohort_page_time.loc[cohort_page_time['time_spent'].idxmax()]

    # Calculate the z-score of the average time spent for each cohort on that page
    page_zscore = df[~df['page'].isin(omit_pages)].groupby(['cohort_id', 'page'])['time_spent_zscore'].mean().reset_index()

    # Filter the z-score DataFrame to only include the cohort with the highest average time spent on the page
    max_zscore = page_zscore[(page_zscore['cohort_id'] == max_cohort_page_time['cohort_id']) & (page_zscore['page'] == max_cohort_page_time['page'])]

    # Check if the z-score is significantly higher than the other cohorts
    if max_zscore['time_spent_zscore'].values[0] > 2:
        print('Cohort {} referred to page {} significantly more than other cohorts'.format(max_cohort_page_time['cohort_id'], max_cohort_page_time['page']))
    else:
        print('No cohort referred to a page significantly more than other cohorts with very minimal and limited interaction and duration.')
        print("This function performs an analysis of the time spent by each cohort on each web page and identifies if any cohort significantly spends more time on a particular web page than others. The function first calculates the time spent z-score and the average time spent by each cohort on each page. Then, it creates a list of pages to omit, which include pages like the homepage, search pages, table of contents, and appendix. Next, it groups the data by cohort_id and page and calculates the average time spent. It then identifies the cohort and page combination with the highest average time spent. After that, the function calculates the z-score of the average time spent for each cohort on that page and filters the z-score DataFrame to only include the cohort with the highest average time spent on the page. If the z-score is significantly higher than the other cohorts, the function prints a message indicating that the identified cohort referred to the identified page significantly more than other cohorts. Otherwise, it prints a message indicating that no cohort referred to a page significantly more than other cohorts with very minimal and limited interaction and duration.")
    
def plot_average_incentive(df):
    """
    Plots a bar chart of the average incentive per team,
    highlighting the top 3 teams in red and others in blue.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the incentive and team columns.
    """
    # Calculate the average incentive per team
    grouped_df = df.groupby('team')['incentive'].mean().reset_index()

    # Calculate the overall average incentive across all teams
    average_incentive = df['incentive'].mean()

    # Find the top 3 teams by their average incentives
    top_3_teams = grouped_df.nlargest(3, 'incentive')['team']

    # Create a custom color list, setting 'red' for top 3 teams and 'blue' for others
    colors = ['grey' if team in top_3_teams.values else 'blue' for team in grouped_df['team']]

    # Plot the bar chart
    plt.bar(grouped_df['team'].astype(str), grouped_df['incentive'], color=colors)
    plt.axhline(average_incentive, color='g', linestyle='--', label='Average Incentive')

    plt.xlabel('Team')
    plt.ylabel('Incentive')
    plt.title('Average Incentive per Team')
    plt.legend()
    plt.show()
              
import matplotlib.pyplot as plt

def plot_top_least_accessed_pages(df):
    """
    Plots the top and least accessed pages for each category.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the access logs.

    Returns:
    --------
    None
    """

    # create a dataframe for the top accessed pages for each category
    top_pages = df.groupby(['category', 'page']).size().reset_index(name='counts')
    top_pages = top_pages.sort_values(['category', 'counts'], ascending=[True, False]).groupby('category').head(3)

    # create a dataframe for the least accessed pages for each category
    least_pages = df.groupby(['category', 'page']).size().reset_index(name='counts')
    least_pages = least_pages.sort_values(['category', 'counts'], ascending=[True, True]).groupby('category').head(3)

    # remove certain pages from the dataframes
    pages_to_remove = ['/', 'search/search_index.json', 'toc', 'spring', 'appendix']
    top_pages = top_pages[~top_pages['page'].isin(pages_to_remove)]
    least_pages = least_pages[~least_pages['page'].isin(pages_to_remove)]

    # create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    # plot the top accessed pages
    for i, (category, data) in enumerate(top_pages.groupby('category')):
        ax[0].barh(data['page'], data['counts'], color=f'C{i}', label=category)
    ax[0].set_xlabel('Number of times accessed')
    ax[0].set_ylabel('Page')
    ax[0].set_title('Top accessed pages')
    ax[0].legend()

    # plot the least accessed pages
    for i, (category, data) in enumerate(least_pages.groupby('category')):
        ax[1].barh(data['page'], data['counts'], color=f'C{i}', label=category)
    ax[1].set_xlabel('Number of times accessed')
    ax[1].set_ylabel('Page')
    ax[1].set_title('Least accessed pages')
    ax[1].legend()

    # adjust the layout and display the chart
    fig.tight_layout()
    print("The plot_top_least_accessed_pages() function plots the top and least accessed pages for each category.")
    plt.show()
    
def plot_actual_productivity_by_team(df):
    # Calculate the average actual productivity across all teams
    avg_actual_productivity = df['actual_productivity'].mean()

    # Find the team with the highest actual productivity
    highest_team = df.loc[df['actual_productivity'].idxmax()]['team']

    # Prepare the data for the bar chart
    team_data = df.groupby('team')['actual_productivity'].mean().reset_index()

    # Create the colormap for the bars
    norm = mcolors.Normalize(vmin=team_data['actual_productivity'].min(), vmax=team_data['actual_productivity'].max())
    cmap = plt.get_cmap('Blues')

    # Create the bar chart
    plt.bar(team_data['team'].astype(str), team_data['actual_productivity'],
            color=cmap(norm(team_data['actual_productivity'].values)))

    # Add horizontal lines for the average actual productivity and the factory productivity standard
    plt.axhline(avg_actual_productivity, color='green', linestyle='--', label='Average Productivity')
    plt.axhline(0.75, color='red', linestyle='--', label='Factory Productivity Standard')

    # Set labels and title
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')
    plt.title('Actual Productivity by Team')
    plt.legend()

    plt.show()
    
    
def plot_actual_productivity_by_team(df):
    # Calculate the average actual productivity across all teams
    avg_actual_productivity = df['actual_productivity'].mean()

    # Find the team with the highest actual productivity
    highest_team = df.loc[df['actual_productivity'].idxmax()]['team']

    # Prepare the data for the bar chart
    team_data = df.groupby('team')['actual_productivity'].mean().reset_index()

    # Create the colormap for the bars
    norm = mcolors.Normalize(vmin=team_data['actual_productivity'].min(), vmax=team_data['actual_productivity'].max())
    cmap = plt.get_cmap('Blues')

    # Create the bar chart
    plt.bar(team_data['team'].astype(str), team_data['actual_productivity'],
            color=cmap(norm(team_data['actual_productivity'].values)))

    # Add horizontal lines for the average actual productivity and the factory productivity standard
    plt.axhline(avg_actual_productivity, color='green', linestyle='--', label='Average Productivity')
    plt.axhline(0.75, color='red', linestyle='--', label='Factory Productivity Standard')

    # Set labels and title
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')
    plt.title('Actual Productivity by Team')
    plt.legend()

    plt.show()
    
    
import matplotlib.colors as mcolors

def team_productivity_chart(df):
    # Calculate the average actual productivity across all teams
    avg_actual_productivity = df['actual_productivity'].mean()

    # Find the team with the highest actual productivity
    highest_team = df.loc[df['actual_productivity'].idxmax()]['team']

    # Prepare the data for the bar chart
    team_data = df.groupby('team')['actual_productivity'].mean().reset_index()

    # Create the colormap for the bars
    norm = mcolors.Normalize(vmin=team_data['actual_productivity'].min(), vmax=team_data['actual_productivity'].max())
    cmap = plt.get_cmap('Blues')

    # Create the bar chart
    plt.bar(team_data['team'].astype(str), team_data['actual_productivity'],
            color=cmap(norm(team_data['actual_productivity'].values)))

    # Add horizontal lines for the average actual productivity and the factory productivity standard
    plt.axhline(avg_actual_productivity, color='green', linestyle='--', label='Average Productivity')
    plt.axhline(0.75, color='red', linestyle='--', label='Factory Productivity Standard')

    # Set labels and title
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')
    plt.title('Actual Productivity by Team')
    plt.legend()

    plt.show()
    
def plot_actual_productivity(df):
    # Calculate the average actual productivity across all teams
    avg_actual_productivity = df['actual_productivity'].mean()

    # Find the team with the highest actual productivity
    highest_team = df.loc[df['actual_productivity'].idxmax()]['team']

    # Prepare the data for the bar chart
    team_data = df.groupby('team')['actual_productivity'].mean().reset_index()

    # Create the colormap for the bars
    norm = mcolors.Normalize(vmin=team_data['actual_productivity'].min(), vmax=team_data['actual_productivity'].max())
    cmap = plt.get_cmap('Blues')

    # Create the bar chart
    plt.bar(team_data['team'].astype(str), team_data['actual_productivity'],
            color=cmap(norm(team_data['actual_productivity'].values)))

    # Add horizontal lines for the average actual productivity and the factory productivity standard
    plt.axhline(avg_actual_productivity, color='green', linestyle='--', label='Average Productivity')
    plt.axhline(0.75, color='red', linestyle='--', label='Factory Productivity Standard')

    # Set labels and title
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')
    plt.title('Actual Productivity by Team')
    plt.legend()

    plt.show()
    
    
def plot_avg_incentive_per_team(df):
    # Calculate the average incentive per team
    grouped_df = df.groupby('team')['incentive'].mean().reset_index()

    # Calculate the overall average incentive across all teams
    average_incentive = df['incentive'].mean()

    # Plot the bar chart
    plt.bar(grouped_df['team'].astype(str), grouped_df['incentive'])
    plt.axhline(average_incentive, color='r', linestyle='--', label='Average Incentive')

    plt.xlabel('Team')
    plt.ylabel('Incentive')
    plt.title('Average Incentive per Team')
    plt.legend()
    plt.show()
    
def productivity_by_team(df):
    # Calculate the average actual productivity
    avg_actual_productivity = df['actual_productivity'].mean()

    # Find the top 3 teams with highest actual productivity
    top_3_actual_productivity_teams = df.groupby('team')['actual_productivity'].mean().sort_values(ascending=False).head(3).index

    # Create a new column to indicate if the team is in the top 3
    df['highlight'] = df['team'].apply(lambda x: 'Top 3' if x in top_3_actual_productivity_teams else 'Others')

    # Set the color palette for the chart
    palette = {'Top 3': 'darkgrey', 'Others': 'lightgrey'}

    # Create the bar plot, highlighting the top 3 teams
    sns.barplot(x='team', y='actual_productivity', data=df, hue='highlight', palette=palette, dodge=False)

    # Add the average actual productivity h-line
    plt.axhline(avg_actual_productivity, color='green', linestyle='--', label=f'Average Actual Productivity: {avg_actual_productivity:.2f}')

    # Set the title and axis labels
    plt.title('Actual Productivity by Team')
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')

    # Rotate the x-axis labels 90 degrees
    plt.xticks(rotation=90)

    # Show the legend
    plt.legend(title='Team Category', loc='upper right')

    # Show the plot
    plt.show()
    
  
    
'''def plot_resampled_data(data, title, chart_type='line'):
    weekly_filtered = weekly[['over_time', 'wip']]
    plt.figure(figsize=(12, 6))
    
    if chart_type == 'line':
        sns.lineplot(data=data)
    elif chart_type == 'bar':
        sns.barplot(data=data)
        
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()

# Filter the columns to include only 'over_time', 'wip', 'idle_time', and 'no_of_workers'
weekly_filtered = weekly[['over_time', 'wip']]

# Plot weekly resampled data with line chart
plot_resampled_data(weekly_filtered, 'Weekly Data', chart_type='line')'''

def plot_overtime(df):
    # Calculate the average overtime across all teams
    avg_overtime = df['over_time'].mean()

    # Find the three teams with the most overtime
    top_3_overtime_teams = df.groupby('team')['over_time'].mean().sort_values(ascending=False).head(3).index

    # Create a new column to indicate if the team is in the top 3
    df['highlight'] = df['team'].apply(lambda x: 'Top 3' if x in top_3_overtime_teams else 'Others')

    plt.figure(figsize=(10, 6))

    # Plot the bar chart, highlighting the top 3 teams
    sns.barplot(x='team', y='over_time', data=df, hue='highlight', palette={'Top 3': 'gray', 'Others': 'blue'}, dodge=False)

    # Add the average overtime h-line
    plt.axhline(avg_overtime, color='green', linestyle='--', label='Average Overtime')

    plt.title('Overtime by Team')
    plt.xlabel('Team')
    plt.ylabel('Overtime (minutes)')
    plt.legend(title='Team Category', loc='upper right')

    plt.show()
    
    
def actual_vs_predicted(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.show()

    
def get_low_activity(df):
    active_students = df.groupby(['user_id', 'cohort_id']).size().reset_index(name='count')
    inactive_students = active_students[active_students['count'] <= 10]

    print("Low Activity Dataframe: The following table shows the user_id, cohort_id and count of students who have accessed the curriculum 10 times or less.")
    return inactive_students.head(5)
    
    
    
    
import matplotlib.pyplot as plt

def top_ten_lessons(top10_webdev, top10_datasci):
    # Create the horizontal bar chart for web development
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top10_webdev['page'], top10_webdev['visits'])
    ax.set_xlabel('Number of Visits')
    ax.set_ylabel('Page')
    ax.set_title('Top 10 Pages for Web Development')

    # Add a print statement providing a synopsis for the web development visualization
    print("Graduate Favorites: Web Development - This visualization shows the top 10 pages for web development, based on the number of visits.")

    plt.show()

    # Create the horizontal bar chart for data science
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top10_datasci['page'], top10_datasci['visits'])
    ax.set_xlabel('Number of Visits')
    ax.set_ylabel('Page')
    ax.set_title('Top 10 Pages for Data Science')

    # Add a print statement providing a synopsis for the data science visualization
    print("Graduate Favorites: Data Science - This visualization shows the top 10 pages for data science, based on the number of visits.")

    plt.show()
     
    
import geoip2.database
import geoip2.errors

def lookup_location_info(df):
    """
    This function takes in a DataFrame that contains a column with IP addresses,
    uses GeoLite2-City database to look up the location information for each IP,
    and returns a new DataFrame with a new column called "location" that contains
    the location information.
    """
    reader = geoip2.database.Reader('GeoLite2-City.mmdb')
    locations = []
    for ip in df['ip']:
        try:
            response = reader.city(ip)
            location = response.country.name
            if response.subdivisions.most_specific.name:
                location = response.subdivisions.most_specific.name + ', ' + location
            locations.append(location)
        except geoip2.errors.AddressNotFoundError:
            locations.append('Suspect')
    
    df['location'] = locations
    print("GEOLite-City: This module takes in a DataFrame as input and adds a 'location' column that corresponds to the location of the IP addresses in the 'ip' column. It accomplishes this by utilizing the MaxMind GeoIP2 Python API and its free GeoLite2 database that maps IP addresses to locations. The function creates a Reader object using the database file and then loops through each IP address in the DataFrame. For each IP address, the function queries the GeoIP2 database and retrieves the country name and subdivision name (if available). It then concatenates the subdivision and country names (in that order) with a comma separator to form the full location string. If the IP address is not found in the database, the location is marked as 'Suspect'.")
    return df
