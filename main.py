import requests
import re, json
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create an empty DataFrame to store all years' data
all_years_df = pd.DataFrame()

for yil in range(2016, 2025):
    url = "https://www.izmitsu.com.tr/teknikveri.php?yil=" + str(yil)
    
    print(f"Fetching data for year {yil}...")
    print("waiting for response...")
    response = requests.get(url, headers=None)
    print("response received")
    html = response.text
    
    # 1. grab the <script> text
    soup = BeautifulSoup(html, "html.parser")
    
    try:
        js = next(s for s in soup.find_all("script") if "series:" in s.text).string
        
        # 2. extract the series array
        m = re.search(r"series\s*:\s*(\[\s*\{[\s\S]*?\}\s*\])", js)
        
        if not m:
            print(f"No data found for year {yil}, skipping...")
            continue
            
        series_txt = m.group(1)
        
        # 3. normalize JS → JSON
        series_txt = series_txt.replace("'", '"')
        
        # quote unquoted keys (name, data, etc.)
        series_txt = re.sub(r'([{\[,]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', series_txt)
        
        # remove any trailing commas before ] or }
        series_txt = re.sub(r",\s*([\]}])", r"\1", series_txt)
        
        # 4. parse
        series = json.loads(series_txt)
        

        
        # Create a proper time series dataframe for this year
        # Generate dates for the entire year
        start_date = datetime(yil, 1, 1)
        
        # Ensure we don't create more dates than we have data points
        num_days = min(len(series[0]["data"]), 366 if yil % 4 == 0 else 365)
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        
        # Create DataFrame with all series
        year_df = pd.DataFrame({
            'Date': dates,
            'Baraj_Seviyesi': series[0]["data"][:num_days],
            'Max_Isletme_Seviyesi': series[1]["data"][:num_days],
            'Min_Isletme_Seviyesi': series[2]["data"][:num_days]
        })
        
        # Set date as index
        year_df.set_index('Date', inplace=True)
        
        # Append to all years DataFrame
        all_years_df = pd.concat([all_years_df, year_df])
        
        # Basic statistics
        print(f"\nBaraj Seviyesi İstatistikleri ({yil}):")
        print(f"Ortalama seviye: {year_df['Baraj_Seviyesi'].mean():.2f} m")
        print(f"Minimum seviye: {year_df['Baraj_Seviyesi'].min():.2f} m")
        print(f"Maximum seviye: {year_df['Baraj_Seviyesi'].max():.2f} m")
        print(f"Standart sapma: {year_df['Baraj_Seviyesi'].std():.2f} m")
        
        # Plot the data for each year
        plt.figure(figsize=(12, 6))
        plt.plot(year_df.index, year_df['Baraj_Seviyesi'], label='Baraj Seviyesi')
        plt.plot(year_df.index, year_df['Max_Isletme_Seviyesi'], 'r--', label='Maksimum İşletme Seviyesi')
        plt.plot(year_df.index, year_df['Min_Isletme_Seviyesi'], 'g--', label='Minimum İşletme Seviyesi')
        plt.title(f'Baraj Su Seviyesi {yil}')
        plt.xlabel('Tarih')
        plt.ylabel('Seviye (m)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'su_seviyesi_grafikleri/baraj_seviyesi_{yil}.png')
        print(f"Grafik 'baraj_seviyesi_{yil}.png' olarak kaydedildi.")
        
    except Exception as e:
        print(f"Error processing data for year {yil}: {str(e)}")

# Sort the combined DataFrame by date
all_years_df.sort_index(inplace=True)

# Save the combined DataFrame as CSV
all_years_df.to_csv('baraj_seviyesi_tum_yillar.csv')
print(f"\nTüm yılların verileri 'baraj_seviyesi_tum_yillar.csv' olarak kaydedildi.")

# Plot combined data for all years
plt.figure(figsize=(15, 8))
plt.plot(all_years_df.index, all_years_df['Baraj_Seviyesi'], label='Baraj Seviyesi')
plt.plot(all_years_df.index, all_years_df['Max_Isletme_Seviyesi'], 'r--', label='Maksimum İşletme Seviyesi')
plt.plot(all_years_df.index, all_years_df['Min_Isletme_Seviyesi'], 'g--', label='Minimum İşletme Seviyesi')
plt.title('Baraj Su Seviyesi (2016-2024)')
plt.xlabel('Tarih')
plt.ylabel('Seviye (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('su_seviyesi_grafikleri/baraj_seviyesi_tum_yillar.png')
print(f"Tüm yılların grafiği 'baraj_seviyesi_tum_yillar.png' olarak kaydedildi.")
