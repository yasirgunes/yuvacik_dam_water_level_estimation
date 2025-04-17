import requests
import re
import json
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create directory for saving graphs if it doesn't exist
os.makedirs('su_seviyesi_grafikleri', exist_ok=True)

def fetch_data(year):
    """
    Fetch dam water level data from the website for a specific year.
    
    Args:
        year (int): The year to fetch data for
        
    Returns:
        dict or None: Parsed JSON data if successful, None otherwise
    """
    url = f"https://www.izmitsu.com.tr/teknikveri.php?yil={year}"
    
    print(f"Fetching data for year {year}...")
    try:
        print("Waiting for response...")
        response = requests.get(url, headers=None, timeout=30)
        print("Response received")
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} for year {year}")
            return None
            
        html = response.text
        return extract_data_from_html(html, year)
        
    except requests.RequestException as e:
        print(f"Network error fetching data for year {year}: {str(e)}")
        return None

def extract_data_from_html(html, year):
    """
    Extract water level data from HTML content.
    
    Args:
        html (str): HTML content from the website
        year (int): The year of the data
        
    Returns:
        dict or None: Parsed JSON data if successful, None otherwise
    """
    try:
        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")
        
        # Find script tag containing data
        script_tag = next((s for s in soup.find_all("script") if "series:" in s.text), None)
        if not script_tag:
            print(f"No data found for year {year}: script tag with 'series:' not found")
            return None
            
        js = script_tag.string
        
        # Extract series array from JavaScript
        match = re.search(r"series\s*:\s*(\[\s*\{[\s\S]*?\}\s*\])", js)
        if not match:
            print(f"No data found for year {year}: couldn't extract series data")
            return None
            
        series_txt = match.group(1)
        
        # Convert JavaScript to valid JSON
        series_txt = normalize_js_to_json(series_txt)
        
        # Parse JSON
        return json.loads(series_txt)
        
    except Exception as e:
        print(f"Error extracting data from HTML for year {year}: {str(e)}")
        return None

def normalize_js_to_json(js_text):
    """
    Convert JavaScript object notation to valid JSON.
    
    Args:
        js_text (str): JavaScript object text
        
    Returns:
        str: Valid JSON string
    """
    # Replace single quotes with double quotes
    json_text = js_text.replace("'", '"')
    
    # Quote unquoted keys (name, data, etc.)
    json_text = re.sub(r'([{\[,]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', json_text)
    
    # Remove trailing commas before ] or }
    json_text = re.sub(r",\s*([\]}])", r"\1", json_text)
    
    return json_text

def create_year_dataframe(series_data, year):
    """
    Create a pandas DataFrame for a specific year's data.
    
    Args:
        series_data (dict): Parsed JSON data containing series information
        year (int): The year of the data
        
    Returns:
        pandas.DataFrame: DataFrame with date index and water level data
    """
    # Generate dates for the year
    start_date = datetime(year, 1, 1)
    
    # Ensure we don't create more dates than we have data points
    days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    num_days = min(len(series_data[0]["data"]), days_in_year)
    
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Create DataFrame with all series
    df = pd.DataFrame({
        'Date': dates,
        'Baraj_Seviyesi': series_data[0]["data"][:num_days],
        'Max_Isletme_Seviyesi': series_data[1]["data"][:num_days],
        'Min_Isletme_Seviyesi': series_data[2]["data"][:num_days]
    })
    
    # Set date as index
    df.set_index('Date', inplace=True)
    
    return df

def print_statistics(df, year):
    """
    Print statistical information about the water level data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing water level data
        year (int): The year of the data
    """
    print(f"\nBaraj Seviyesi İstatistikleri ({year}):")
    print(f"Ortalama seviye: {df['Baraj_Seviyesi'].mean():.2f} m")
    print(f"Minimum seviye: {df['Baraj_Seviyesi'].min():.2f} m")
    print(f"Maximum seviye: {df['Baraj_Seviyesi'].max():.2f} m")
    print(f"Standart sapma: {df['Baraj_Seviyesi'].std():.2f} m")
    print(f"Veri sayısı: {len(df)}")

def plot_year_data(df, year):
    """
    Create and save a plot for a specific year's water level data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing water level data
        year (int): The year of the data
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Baraj_Seviyesi'], label='Baraj Seviyesi')
    plt.plot(df.index, df['Max_Isletme_Seviyesi'], 'r--', label='Maksimum İşletme Seviyesi')
    plt.plot(df.index, df['Min_Isletme_Seviyesi'], 'g--', label='Minimum İşletme Seviyesi')
    plt.title(f'Baraj Su Seviyesi {year}')
    plt.xlabel('Tarih')
    plt.ylabel('Seviye (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = f'su_seviyesi_grafikleri/baraj_seviyesi_{year}.png'
    plt.savefig(filename)
    print(f"Grafik '{filename}' olarak kaydedildi.")

def plot_all_years_data(all_df, start_year, end_year):
    """
    Create and save a plot for all years' water level data.
    
    Args:
        all_df (pandas.DataFrame): DataFrame containing all years' water level data
        start_year (int): First year in the dataset
        end_year (int): Last year in the dataset
    """
    plt.figure(figsize=(15, 8))
    plt.plot(all_df.index, all_df['Baraj_Seviyesi'], label='Baraj Seviyesi')
    plt.plot(all_df.index, all_df['Max_Isletme_Seviyesi'], 'r--', label='Maksimum İşletme Seviyesi')
    plt.plot(all_df.index, all_df['Min_Isletme_Seviyesi'], 'g--', label='Minimum İşletme Seviyesi')
    plt.title(f'Baraj Su Seviyesi ({start_year}-{end_year})')
    plt.xlabel('Tarih')
    plt.ylabel('Seviye (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = f'su_seviyesi_grafikleri/baraj_seviyesi_tum_yillar.png'
    plt.savefig(filename)
    print(f"Tüm yılların grafiği '{filename}' olarak kaydedildi.")

def main():
    """
    Main function to orchestrate the data collection, analysis and visualization.
    """
    # Define year range
    start_year = 2016
    end_year = 2025
    
    # Create an empty DataFrame to store all years' data
    all_years_df = pd.DataFrame()
    
    # Process each year
    for year in range(start_year, end_year + 1):
        # Fetch and extract data
        series_data = fetch_data(year)
        
        if not series_data:
            print(f"Skipping year {year} due to missing data")
            continue
        
        # Create DataFrame for this year
        year_df = create_year_dataframe(series_data, year)
        
        # Add to combined DataFrame
        all_years_df = pd.concat([all_years_df, year_df])
        
        # Print statistics
        print_statistics(year_df, year)
        
        # Create and save plot
        plot_year_data(year_df, year)
    
    # Sort the combined DataFrame by date
    all_years_df.sort_index(inplace=True)
    
    # Save the combined DataFrame as CSV
    csv_filename = 'baraj_seviyesi_tum_yillar.csv'
    all_years_df.to_csv(csv_filename)
    print(f"\nTüm yılların verileri '{csv_filename}' olarak kaydedildi.")
    
    # Create and save combined plot
    plot_all_years_data(all_years_df, start_year, end_year)
    
    # Additional analysis
    print("\nTüm Yıllar İçin Özet İstatistikler:")
    print(all_years_df['Baraj_Seviyesi'].describe())
    
    return all_years_df

if __name__ == "__main__":
    # Execute main function when script is run directly
    df = main()
