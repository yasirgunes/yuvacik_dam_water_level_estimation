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

def create_complete_year_dataframe(df, year):
    """
    Create a complete year DataFrame with all days, filling missing values with interpolation.
    
    Args:
        df (pandas.DataFrame): Original DataFrame with possibly missing dates
        year (int): The year to create complete data for
        
    Returns:
        pandas.DataFrame: Complete DataFrame with all days in the year, gaps filled with interpolation
    """
    # Create a complete date range for the year
    start_date = datetime(year, 1, 1)
    days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    
    # For the current year, only include dates up to today
    if year == datetime.now().year:
        end_date = datetime.now()
        if end_date.year > year:
            end_date = datetime(year, 12, 31)
    else:
        end_date = datetime(year, 12, 31)
        
    # Create the complete date range
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a new DataFrame with all dates
    complete_df = pd.DataFrame(index=full_date_range)
    
    # Join with original data
    complete_df = complete_df.join(df)
    
    # Check for missing data before interpolation
    missing_days = complete_df['Baraj_Seviyesi'].isna().sum()
    total_days = len(complete_df)
    
    # Report on missing data
    if missing_days > 0:
        print(f"Veri kalitesi: {year} yılında {missing_days} gün eksik veri var ({missing_days/total_days*100:.1f}%)")
        print(f"Eksik veriler interpolasyon ile doldurulacak.")
    
    # Fill missing values using interpolation
    # First for Baraj_Seviyesi
    complete_df['Baraj_Seviyesi'] = complete_df['Baraj_Seviyesi'].interpolate(method='linear')
    
    # For reference levels, use forward fill and backward fill since these should be relatively constant
    # Using ffill() and bfill() instead of fillna(method='ffill') to avoid deprecation warnings
    complete_df['Max_Isletme_Seviyesi'] = complete_df['Max_Isletme_Seviyesi'].ffill().bfill()
    complete_df['Min_Isletme_Seviyesi'] = complete_df['Min_Isletme_Seviyesi'].ffill().bfill()
    
    return complete_df

def check_data_quality(df):
    """
    Check data quality and return information about missing or problematic data.
    
    Args:
        df (pandas.DataFrame): DataFrame to check
        
    Returns:
        dict: Data quality information
    """
    # Check for missing values
    missing_values = df.isna().sum()
    
    # Check for outliers using IQR method
    Q1 = df['Baraj_Seviyesi'].quantile(0.25)
    Q3 = df['Baraj_Seviyesi'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['Baraj_Seviyesi'] < lower_bound) | (df['Baraj_Seviyesi'] > upper_bound)]
    
    # Check for sudden changes (potential errors)
    df_temp = df.copy()
    df_temp['Baraj_Seviyesi_diff'] = df_temp['Baraj_Seviyesi'].diff().abs()
    
    # Average daily change
    avg_change = df_temp['Baraj_Seviyesi_diff'].mean()
    std_change = df_temp['Baraj_Seviyesi_diff'].std()
    
    # Looking for changes more than 3 standard deviations from mean
    threshold = avg_change + 3 * std_change
    sudden_changes = df_temp[df_temp['Baraj_Seviyesi_diff'] > threshold]
    
    return {
        'missing_values': missing_values,
        'missing_days': missing_values['Baraj_Seviyesi'] if 'Baraj_Seviyesi' in missing_values else 0,
        'outliers': len(outliers),
        'outlier_dates': outliers.index.tolist(),
        'sudden_changes': len(sudden_changes),
        'sudden_change_dates': sudden_changes.index.tolist(),
        'avg_daily_change': avg_change,
        'max_daily_change': df_temp['Baraj_Seviyesi_diff'].max()
    }

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
    
    # Check for data gaps
    days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    
    # For current year, adjust expected days
    if year == datetime.now().year:
        days_in_year = (datetime.now() - datetime(year, 1, 1)).days + 1
    
    if len(df) < days_in_year:
        print(f"Uyarı: {year} yılı için {days_in_year} gün olması gerekirken sadece {len(df)} gün veri var.")
        print(f"      {days_in_year - len(df)} gün veri eksik ({(days_in_year - len(df))/days_in_year*100:.1f}%)")

def plot_year_data(df, year, with_gaps=False):
    """
    Create and save a plot for a specific year's water level data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing water level data
        year (int): The year of the data
        with_gaps (bool): Whether to show gaps in the data
    """
    plt.figure(figsize=(12, 6))
    
    if with_gaps:
        # Plot with visible gaps for missing data
        plt.plot(df.index, df['Baraj_Seviyesi'], label='Baraj Seviyesi', marker='o', markersize=2)
    else:
        # Plot with interpolated data to fill small gaps
        plt.plot(df.index, df['Baraj_Seviyesi'].interpolate(method='linear'), label='Baraj Seviyesi')
    
    plt.plot(df.index, df['Max_Isletme_Seviyesi'], 'r--', label='Maksimum İşletme Seviyesi')
    plt.plot(df.index, df['Min_Isletme_Seviyesi'], 'g--', label='Minimum İşletme Seviyesi')
    plt.title(f'Baraj Su Seviyesi {year}')
    plt.xlabel('Tarih')
    plt.ylabel('Seviye (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Add text for data quality
    days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    if year == datetime.now().year:
        days_in_year = (datetime.now() - datetime(year, 1, 1)).days + 1
        
    data_percentage = len(df) / days_in_year * 100
    
    plt.figtext(0.02, 0.02, f'Veri kalitesi: {data_percentage:.1f}% ({len(df)}/{days_in_year} gün)', 
                fontsize=9, color='navy')
    
    filename = f'su_seviyesi_grafikleri/baraj_seviyesi_{year}.png'
    plt.savefig(filename, dpi=300)
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
    
    # Fill small gaps for plotting
    plot_df = all_df.copy()
    plot_df['Baraj_Seviyesi'] = plot_df['Baraj_Seviyesi'].interpolate(method='linear', limit=3)
    
    plt.plot(plot_df.index, plot_df['Baraj_Seviyesi'], label='Baraj Seviyesi')
    plt.plot(plot_df.index, plot_df['Max_Isletme_Seviyesi'], 'r--', label='Maksimum İşletme Seviyesi')
    plt.plot(plot_df.index, plot_df['Min_Isletme_Seviyesi'], 'g--', label='Minimum İşletme Seviyesi')
    plt.title(f'Baraj Su Seviyesi ({start_year}-{end_year})')
    plt.xlabel('Tarih')
    plt.ylabel('Seviye (m)')
    plt.legend()
    plt.grid(True)
    
    # Add year dividers and labels
    for year in range(start_year + 1, end_year + 1):
        plt.axvline(x=datetime(year, 1, 1), color='gray', linestyle='-', alpha=0.3)
        plt.text(datetime(year, 1, 15), plot_df['Baraj_Seviyesi'].min(), str(year), 
                 fontsize=9, color='gray', rotation=90, va='bottom')
    
    # Display completion percentage
    total_days = (datetime(end_year, 12, 31) - datetime(start_year, 1, 1)).days + 1
    if end_year == datetime.now().year:
        total_days = (datetime.now() - datetime(start_year, 1, 1)).days + 1
    
    data_percentage = len(plot_df.dropna(subset=['Baraj_Seviyesi'])) / total_days * 100
    plt.figtext(0.02, 0.02, f'Toplam veri kalitesi: {data_percentage:.1f}% ({len(plot_df.dropna(subset=["Baraj_Seviyesi"]))}/{total_days} gün)', 
                fontsize=9, color='navy')
    
    plt.tight_layout()
    
    filename = f'su_seviyesi_grafikleri/baraj_seviyesi_tum_yillar.png'
    plt.savefig(filename, dpi=300)
    print(f"Tüm yılların grafiği '{filename}' olarak kaydedildi.")
    
    # Create yearly averages plot
    create_yearly_averages_plot(all_df, start_year, end_year)

def create_yearly_averages_plot(df, start_year, end_year):
    """
    Create a plot showing yearly averages for easier comparison.
    
    Args:
        df (pandas.DataFrame): DataFrame with all data
        start_year (int): First year in the dataset
        end_year (int): Last year in the dataset
    """
    # Create resample by year
    yearly_avg = df.groupby(pd.Grouper(freq='YE')).mean()
    yearly_min = df.groupby(pd.Grouper(freq='YE')).min()
    yearly_max = df.groupby(pd.Grouper(freq='YE')).max()
    
    # Count data points per year
    data_counts = df.groupby(pd.Grouper(freq='YE')).count()
    
    # Create figure with adjusted size and margins
    plt.figure(figsize=(14, 7))
    
    # Bar chart for yearly averages
    years = [year for year in range(start_year, end_year + 1)]
    available_years = [year.year for year in yearly_avg.index]
    
    avgs = [yearly_avg.loc[datetime(year, 12, 31), 'Baraj_Seviyesi'] 
            if year in available_years and datetime(year, 12, 31) in yearly_avg.index else np.nan 
            for year in years]
    
    mins = [yearly_min.loc[datetime(year, 12, 31), 'Baraj_Seviyesi'] 
            if year in available_years and datetime(year, 12, 31) in yearly_min.index else np.nan 
            for year in years]
    
    maxs = [yearly_max.loc[datetime(year, 12, 31), 'Baraj_Seviyesi'] 
            if year in available_years and datetime(year, 12, 31) in yearly_max.index else np.nan 
            for year in years]
    
    # Create bar chart with error bars
    bars = plt.bar(years, avgs, yerr=[np.array(avgs)-np.array(mins), np.array(maxs)-np.array(avgs)],
                   alpha=0.7, capsize=10, color='skyblue', label='Ortalama Seviye')
    
    # Add data point counts on top of bars
    for i, year in enumerate(years):
        if year in available_years and not np.isnan(avgs[i]):
            try:
                count = data_counts.loc[datetime(year, 12, 31), 'Baraj_Seviyesi']
                days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
                if year == datetime.now().year:
                    days_in_year = (datetime.now() - datetime(year, 1, 1)).days + 1
                percentage = count / days_in_year * 100
                plt.text(i, avgs[i] + 2, f"{count}/{days_in_year}\n({percentage:.1f}%)", 
                         ha='center', va='bottom', fontsize=8)
            except (KeyError, TypeError) as e:
                print(f"Warning: Could not add data count for year {year}: {e}")
    
    plt.axhline(y=df['Max_Isletme_Seviyesi'].mean(), color='r', linestyle='--', 
                label=f"Max İşletme Seviyesi ({df['Max_Isletme_Seviyesi'].mean():.2f}m)")
    plt.axhline(y=df['Min_Isletme_Seviyesi'].mean(), color='g', linestyle='--', 
                label=f"Min İşletme Seviyesi ({df['Min_Isletme_Seviyesi'].mean():.2f}m)")
    
    plt.title('Yıllık Ortalama Baraj Su Seviyeleri')
    plt.xlabel('Yıl')
    plt.ylabel('Ortalama Seviye (m)')
    plt.xticks(years)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Adjust layout with explicit parameters instead of tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    
    filename = 'su_seviyesi_grafikleri/yillik_ortalama_seviyeler.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Yıllık ortalama seviyeleri grafiği '{filename}' olarak kaydedildi.")

def analyze_seasonal_patterns(df):
    """
    Analyze seasonal patterns in the data.
    
    Args:
        df (pandas.DataFrame): DataFrame with all data
    """
    # Create month averages
    df_copy = df.copy()
    df_copy['Month'] = df_copy.index.month
    monthly_avg = df_copy.groupby('Month')['Baraj_Seviyesi'].mean()
    monthly_std = df_copy.groupby('Month')['Baraj_Seviyesi'].std()
    
    plt.figure(figsize=(14, 7))
    
    # Bar chart for monthly averages
    months = list(range(1, 13))
    month_names = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 
                   'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
    
    plt.bar(months, monthly_avg, yerr=monthly_std, alpha=0.7, capsize=5, color='skyblue')
    
    plt.title('Aylık Ortalama Baraj Su Seviyeleri')
    plt.xlabel('Ay')
    plt.ylabel('Ortalama Seviye (m)')
    plt.xticks(months, month_names, rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Adjust layout with explicit parameters
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    
    filename = 'su_seviyesi_grafikleri/aylik_ortalama_seviyeler.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Aylık ortalama seviyeleri grafiği '{filename}' olarak kaydedildi.")
    
    # Calculate highest/lowest month only if we have data for all months
    highest_month = month_names[monthly_avg.idxmax()-1] if not monthly_avg.empty else "Veri yetersiz"
    lowest_month = month_names[monthly_avg.idxmin()-1] if not monthly_avg.empty else "Veri yetersiz"
    annual_range = monthly_avg.max() - monthly_avg.min() if not monthly_avg.empty else 0
    
    return {
        'monthly_avg': monthly_avg,
        'highest_month': highest_month,
        'lowest_month': lowest_month,
        'annual_range': annual_range
    }

def main():
    """
    Main function to orchestrate the data collection, analysis and visualization.
    """
    # Define year range
    start_year = 2016
    end_year = 2025
    
    # Create an empty DataFrame to store all years' data
    all_years_df = pd.DataFrame()
    all_years_complete_df = pd.DataFrame()
    data_quality_info = {}
    
    # Process each year
    for year in range(start_year, end_year + 1):
        # Fetch and extract data
        series_data = fetch_data(year)
        
        if not series_data:
            print(f"Skipping year {year} due to missing data")
            continue
        
        # Create DataFrame for this year
        year_df = create_year_dataframe(series_data, year)
        
        # Create complete DataFrame with missing values properly marked
        complete_year_df = create_complete_year_dataframe(year_df, year)
        
        # Add to combined DataFrame
        all_years_df = pd.concat([all_years_df, year_df])
        all_years_complete_df = pd.concat([all_years_complete_df, complete_year_df])
        
        # Check data quality
        data_quality_info[year] = check_data_quality(year_df)
        
        # Print statistics
        print_statistics(year_df, year)
        
        # Create and save plot
        plot_year_data(year_df, year)
    
    # Sort the combined DataFrame by date
    all_years_df.sort_index(inplace=True)
    all_years_complete_df.sort_index(inplace=True)
    
    # Save the combined DataFrame as CSV
    csv_filename = 'baraj_seviyesi_tum_yillar.csv'
    all_years_df.to_csv(csv_filename)
    print(f"\nTüm yılların verileri '{csv_filename}' olarak kaydedildi.")
    
    # Save the complete DataFrame with gaps filled by interpolation
    complete_csv_filename = 'baraj_seviyesi_tum_yillar_eksiksiz.csv'
    
    # Check if there are any remaining NaN values
    missing_count_before = all_years_complete_df['Baraj_Seviyesi'].isna().sum()
    if missing_count_before > 0:
        print(f"Uyarı: Tam veri setinde hala {missing_count_before} eksik değer var. Son bir interpolasyon uygulanıyor.")
        # Final interpolation to ensure no gaps remain
        all_years_complete_df = all_years_complete_df.interpolate(method='linear')
        # Use ffill and bfill instead of fillna(method='...')
        all_years_complete_df = all_years_complete_df.ffill().bfill()
    
    # Verify no missing values remain
    missing_count_after = all_years_complete_df['Baraj_Seviyesi'].isna().sum()
    if missing_count_after == 0:
        print(f"Başarılı: Tam veri seti tüm boşluklar doldurularak oluşturuldu.")
    else:
        print(f"Uyarı: Tam veri setinde hala {missing_count_after} eksik değer var.")
    
    all_years_complete_df.to_csv(complete_csv_filename)
    print(f"Boşlukları interpolasyon ile tamamlanmış veri seti '{complete_csv_filename}' olarak kaydedildi.")
    
    # Create and save combined plot
    plot_all_years_data(all_years_df, start_year, end_year)
    
    # Additional analysis
    print("\nTüm Yıllar İçin Özet İstatistikler:")
    print(all_years_df['Baraj_Seviyesi'].describe())
    
    # Data quality summary
    print("\nVeri Kalitesi Özeti:")
    total_days = sum(366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365 
                     for year in data_quality_info.keys() if year < datetime.now().year)
    
    # Add days for current year
    if datetime.now().year in data_quality_info:
        total_days += (datetime.now() - datetime(datetime.now().year, 1, 1)).days + 1
    
    total_data_points = sum(len(all_years_df[all_years_df.index.year == year]) 
                           for year in data_quality_info.keys())
    
    print(f"Toplam veri gün sayısı: {total_data_points}/{total_days} ({total_data_points/total_days*100:.1f}%)")
    
    total_outliers = sum(info['outliers'] for info in data_quality_info.values())
    print(f"Toplam aykırı değer sayısı: {total_outliers}")
    
    total_sudden_changes = sum(info['sudden_changes'] for info in data_quality_info.values())
    print(f"Toplam ani değişim sayısı: {total_sudden_changes}")
    
    # Seasonal analysis
    seasonal_info = analyze_seasonal_patterns(all_years_df)
    print("\nMevsimsel Analiz:")
    print(f"En yüksek su seviyesi ayı: {seasonal_info['highest_month']}")
    print(f"En düşük su seviyesi ayı: {seasonal_info['lowest_month']}")
    print(f"Yıllık ortalama değişim aralığı: {seasonal_info['annual_range']:.2f} m")
    
    return all_years_df, all_years_complete_df, data_quality_info

if __name__ == "__main__":
    # Execute main function when script is run directly
    df, complete_df, quality_info = main()
