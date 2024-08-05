import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def parse_full_credits(movie_directory):
    """
    Parse the full credits page of a given movie to get the list of actors and their movies/TV shows.
    
    Parameters:
    movie_directory (str): The directory string for the movie on TMDB.
    
    Returns:
    pd.DataFrame: DataFrame containing actor names and movie/TV show titles.
    """
    base_url = "https://www.themoviedb.org/movie/"
    full_cast_url = f"{base_url}{movie_directory}/cast"
    
    # Request the full cast page of the movie
    response = requests.get(full_cast_url, headers={'User-Agent': 'Mozilla/5.0'})

    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Initialize an empty DataFrame to store the actor names and their movies/TV shows
    df = pd.DataFrame(columns=['actor', 'movie_or_TV_name'])

    # Select all actor links in the cast section, excluding crew members
    # 'ol.people li:not(:has(p.episode_count_crew))' selects list items without 'p.episode_count_crew'
    # 'a[href*="/person/"]' selects anchor tags with 'href' attribute containing '/person/'
    actor_links = soup.select('ol.people li:not(:has(p.episode_count_crew)) a[href*="/person/"]')
    
    # Iterate over each selected actor link
    for a_tag in actor_links:
        # Extract the actor's directory from the href attribute
        actor_url_suffix = a_tag['href']
        actor_directory = actor_url_suffix.split('/')[-1]
        
        # Parse the actor's page to get their movies/TV shows and update the DataFrame
        df = parse_actor_page(df, actor_directory)
        
        # Wait for a short period between requests to prevent being blocked by the website
        time.sleep(0.6) 
        
    # Remove duplicate entries from the DataFrame
    df.drop_duplicates(inplace=True)
    
    # Sort the DataFrame by actor names and movie/TV show titles
    df.sort_values(by=['actor', 'movie_or_TV_name'], inplace=True)
    
    return df

def parse_actor_page(df, actor_directory):
    """
    Parse the actor's page to get the list of movies and TV shows they have been in.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to append the actor and their movies/TV shows to.
    actor_directory (str): The directory string for the actor on TMDB.
    
    Returns:
    pd.DataFrame: Updated DataFrame with the actor and their movies/TV shows.
    """
    base_url = "https://www.themoviedb.org/person/"
    actor_url = f"{base_url}{actor_directory}?credit_department=Acting"
    
    # Request the actor's page
    response = requests.get(actor_url, headers={'User-Agent': 'Mozilla/5.0'}) 
    
    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Get the actor's name from the 'h2' tag with class 'title'
    actor_name_tag = soup.find('h2', class_='title')
    actor_name = actor_name_tag.text.strip() if actor_name_tag else 'Unknown Actor'

    # Find the acting section by searching for the section with 'Acting' in the 'h3' header
    acting_section = None
    for section in soup.find_all('section'):
        header = section.find('h3')
        if header and 'Acting' in header.text:
            acting_section = section
            break

    # If the acting section is found, extract the movie and TV show titles
    if acting_section:
        titles_set = set()  # Use a set to avoid duplicate titles
        roles = acting_section.find_all('bdi')  # Find all 'bdi' tags containing titles
        for role in roles:
            title = role.text.strip()
            if title not in titles_set:
                titles_set.add(title)  # Add unique titles to the set
                # Append the actor and movie/TV show to the DataFrame
                df = pd.concat([df, pd.DataFrame({'actor': [actor_name], 'movie_or_TV_name': [title]})], ignore_index=True)

    return df
