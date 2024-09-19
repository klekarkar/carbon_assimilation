import xarray as xr
import numpy as np
from xarray import DataArray
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def compute_standardized_anomaly(ds: DataArray) -> DataArray:
    """
    Compute the standardized anomaly of a DataArray.
    The standardized anomaly of each month with respect to all the corresponding months in the time series.
    For each month, the standardized anomaly is calculated as the anomaly divided by the standard deviation of the anomaly.
    Parameters
    ----------
    ds : DataArray
        The input rainfall data. At daily temporal scale.
    Returns
    -------
    DataArray
        The standardized anomaly.
    """
    # Step 1: Compute monthly total rainfall
    monthly_mean = ds.resample(time='ME').sum('time')

    # Step 2: Compute mean of each month across all years
    monthly_mean_grouped = monthly_mean.groupby('time.month').mean()

     # Step 3: Compute monthly anomalies
    # vectorized more efficient method
    ds_anomalies = monthly_mean.groupby('time.month') - monthly_mean_grouped

    # Step 4: Calculate the standard deviation of the anomalies for each month
    # Group anomalies by month and compute standard deviation over the time dimension
    anomalies_stdev_monthly = ds_anomalies.groupby('time.month').std()
    
    #compute the standardized monthly anomalies
    #Divide each monthly anomaly by the standard deviation of the corresponding month to get the standardized anomalies.
    standardized_anomalies = ds_anomalies.groupby('time.month') / anomalies_stdev_monthly

    return standardized_anomalies

#####################################################################

def pseudo_z_scores(ds):
    """
    Indices with respect to deviations from multiyear avarage
    Pseudo z-scores, normalized to a mean of 0 and a standard deviation of 1
    Compute mean and std dev of each pixel and calculate standardized deviation from mean 
    Parameters
    ----------
    ds : DataArray
        The input data array.
    Returns
    -------
    DataArray
        The standardized anomaly.
    
    """
    #compute the standardized anomalies
    mean = ds.mean(dim='time')

    #standard deviation of the rainfall
    std_dev = ds.std(dim='time')

    #compute the standardized anomalies
    z_anomalies = (ds - mean) / std_dev

    return z_anomalies

#####################################################################
def regression_scatter_plot(ds1, ds2, min_lon, max_lon, min_lat, max_lat,start_date, end_date):
    """
    Create a scatter plot of two datasets and a linear regression line.
    Inputs:
    ds1 (xarray.Dataset): The first dataset to compare.
    ds2 (xarray.Dataset): The second dataset to compare.
    min_lon (float): The minimum longitude for the region to compare.
    max_lon (float): The maximum longitude for the region to compare.
    min_lat (float): The minimum latitude for the region to compare.
    max_lat (float): The maximum latitude for the region to compare.
    start_date (str): The start date for the time series.
    end_date (str): The end date for the time series.

    Returns:
    Regression plot of the two datasets.
    """
    # Determine correct slicing order for latitude in ds1
    if ds1.lat[0] < ds1.lat[-1]:  # Latitude increasing
        ds1_lat_slice = slice(min_lat, max_lat)
    else:  # Latitude decreasing
        ds1_lat_slice = slice(max_lat, min_lat)

    # Select and average over the specified region in ds1
    ds1_ts = ds1.sel(lon=slice(min_lon, max_lon), lat=ds1_lat_slice).mean(dim=['lat', 'lon'])
    ds1_df = ds1_ts.to_dataframe()

    # Determine correct slicing order for latitude in ds2
    if ds2.lat[0] < ds2.lat[-1]:  # Latitude increasing
        ds2_lat_slice = slice(min_lat, max_lat)
    else:  # Latitude decreasing
        ds2_lat_slice = slice(max_lat, min_lat)

    # Select and average over the specified region in ds2
    ds2_ts = ds2.sel(lon=slice(min_lon, max_lon), lat=ds2_lat_slice).mean(dim=['lat', 'lon'])
    ds2_df = ds2_ts.to_dataframe()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # combine the two dataframes
    combined_df = pd.concat([ds1_df, ds2_df], axis=1)

    x=combined_df.iloc[:, 0][start_date:end_date].values.reshape(-1, 1)
    y=combined_df.iloc[:, 1][start_date:end_date].values.reshape(-1, 1)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Create a linear regression model
    model = LinearRegression()
    model.fit(x, y)

    # Make predictions
    y_pred = model.predict(x)

    #plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color='blue', label='Observed')
    ax.plot(x, y_pred, color='red', label='R$^2$ = {:.2f}'.format(model.score(x, y)))
    ax.set_xlabel(list(ds1.keys())[0])
    ax.set_ylabel(list(ds2.keys())[0])
    ax.legend()

    # Show plot
    plt.show()


import xarray as xr
import numpy as np

def extract_grouped_data(modis_data, dataset_name, agg_func='mean'):
    """
    Extract grouped data to match the temporal resolution of MODIS datasets.
    
    Inputs:
    modis_data: xarray.Dataset with MODIS data at 8-day resolution
    dataset_name: xarray.Dataset with daily data to be resampled to match MODIS resolution
    agg_func: string or callable to aggregate data (e.g., 'mean', 'sum', 'median')
    
    Returns:
    xarray.Dataset with resolution matching the MODIS dataset
    """
    # Define a mapping from string to xarray aggregation method
    agg_funcs = {
        'mean': lambda x: x.mean(dim='time'),
        'sum': lambda x: x.sum(dim='time')
    }

    # Check if agg_func is a string and map to corresponding function
    if isinstance(agg_func, str):
        if agg_func not in agg_funcs:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")
        agg_func = agg_funcs[agg_func]
    
    modis_dates = modis_data.time.values
    modis_date_diff = np.diff(modis_dates).astype('timedelta64[D]').astype(int)

    grouped_values = []
    for date, diff in zip(modis_dates[1:], modis_date_diff):
        
        # Calculate the difference between consecutive dates
        delta = np.timedelta64(diff, 'D')
        start_date = date - (delta - 1)
        end_date = date  # The end date is the current date

        # Select data within the period
        data = dataset_name.sel(time=slice(start_date, end_date))
        
        # Apply the aggregation function
        if callable(agg_func):
            data_agg = agg_func(data)
        else:
            raise ValueError(f"Aggregation function is not callable: {agg_func}")

        # Assign time as an index
        data_agg.coords['time'] = date

        grouped_values.append(data_agg)

    # Combine the aggregated values into a single dataset
    grouped_data = xr.concat(grouped_values, dim='time')

    return grouped_data



def extract_region_timeseries(dataset:xr.Dataset, min_lon: float, max_lon: float,
                               min_lat: float, max_lat: float) -> pd.DataFrame:
    """Extract the timeseries for a specific region from an xarray.Dataset

    Inputs:
    dataset: xarray.Dataset with the data
    min_lon, max_lon, min_lat, max_lat: float, the minimum and maximum longitude and latitude values for the region

    Returns:
    pandas.DataFrame with the timeseries for the specified region
    """

     # Determine correct slicing order for latitude in ds1
    if dataset.lat[0] < dataset.lat[-1]:  # Latitude increasing
        lat_slice = slice(min_lat, max_lat)
    else:  # Latitude decreasing
        lat_slice = slice(max_lat, min_lat)

    # Select and average over the specified region in ds1
    region_timeseries = dataset.sel(lon=slice(min_lon, max_lon), lat=lat_slice).mean(dim=['lat', 'lon'])
    region_df = region_timeseries.to_dataframe()

    return region_df

#-------------------------------------------------------------------------------------------------------------------


#function to extract correlation at different lag periods
def lag_correlation(climate_df: pd.DataFrame, vegetation_df: pd.DataFrame, lags: range) -> pd.DataFrame:

    """Calculate the correlation between soil moisture and vegetation at different lag periods
    Inputs:
    lags: list of lag periods to test
    soil_data_set: xarray.Dataset with soil moisture data
    vegetation_dataset: xarray.Dataset with vegetation process data
    
    Returns:
    pandas.DataFrame with correlation values for each lag period
    """
    # Create an empty list to store the correlation values
    corr_values = []
    p_values=[]
    
    # Iterate over each lag period
    for lag in lags:
        # Shift the vegetation dataset by the lag period
        vegetation_lagged = vegetation_df.shift(time=lag).dropna()

        #align the datasets to get rid of nan values
        climate_df_lagged = climate_df[vegetation_lagged.index[0]:vegetation_lagged.index[-1]]
        
        # Calculate the correlation between the lagged vegetation and soil moisture datasets
        corr, p_value = pearsonr(climate_df_lagged.values.flatten(), vegetation_lagged.values.flatten())
        
        # Append the correlation and p-value to their respective lists
        corr_values.append(corr)
        p_values.append(p_value)
    
    # Create a DataFrame from the correlation values
    correlation_df = pd.DataFrame({
        'Correlation': corr_values,
        'p_value': p_values
    }, index=lags)
    
    return correlation_df

#-------------------------------------------------------------------------------------------------------------------
def detect_season_onset(rainfall_df_daily, threshold_rainy_days=14, cutoff_day=45):
    """ 
    Detect the onset of the rainy season based on accumulated rainfall.
    The onset of the rainy season is defined as the first day with at least 14 consecutive days of positive accumulated rainfall difference.
    Parameters
    ----------
    rainfall_df_daily : pd.DataFrame
        Daily rainfall data.
    threshold_rainy_days : int, optional
        Minimum number of consecutive rainy days to define the onset of the rainy season, by default 14.
    cutoff_day : int, optional
        Day of the year after which the rainy season onset is considered, by default 45.
    Returns
    -------
    pd.DataFrame
        Accumulated rainfall difference for each year.
    pd.Series
        Rainy season onset day for each year.    
     
    """
    rainfall_accumulation = []
    year_onset = {}

    # Calculate mean daily rainfall for the entire dataset
    mean_daily_rainfall = rainfall_df_daily.mean().iloc[0]

    # Loop through each year in the dataset
    for yr in rainfall_df_daily.index.year.unique():
        # Filter data for the current year
        year_data = rainfall_df_daily[rainfall_df_daily.index.year == yr]
        
        if year_data.empty:
            print(f"No data available for year {yr}")
            continue
        
        # Calculate actual accumulated daily rainfall
        actual_accumulated = year_data.cumsum()
        actual_accumulated.index = actual_accumulated.index.dayofyear
        actual_accumulated.columns = ['actual_accumulated']

        # Calculate the difference between actual and average accumulated rainfall
        accumulated_diff = actual_accumulated['actual_accumulated'] - mean_daily_rainfall * actual_accumulated.index
        accumulated_diff.name = yr
        rainfall_accumulation.append(accumulated_diff)

        # Find consecutive days with positive accumulated rainfall difference
        slope = accumulated_diff.diff().rolling(window=threshold_rainy_days).mean()
        positive_slope = slope >= 0.0
        window = positive_slope.rolling(window=threshold_rainy_days).sum()

        # Find the index of the first True value within the window
        rainy_season_start_index = window[window == threshold_rainy_days].index - (threshold_rainy_days - 1)
        rainy_season_start_index = rainy_season_start_index[rainy_season_start_index >= cutoff_day]

        if not rainy_season_start_index.empty:
            rainy_season_start = rainy_season_start_index[0]
            year_onset[yr] = rainy_season_start
            
        else:
            year_onset[yr] = np.nan
        
    year_onset_df = pd.Series(year_onset)
    onset_date_df = year_onset_df.reset_index()
    onset_date_df.columns = ['year', 'dayofyear']
    onset_date_df['date'] = pd.to_datetime(onset_date_df['year'] * 1000 + onset_date_df['dayofyear'], format='%Y%j')

    # Combine all years' accumulated differences into a single DataFrame if needed
    rainfall_accumulation_df = pd.concat(rainfall_accumulation, axis=1)
    return rainfall_accumulation_df, onset_date_df


#-------------------------------------------------------------------------------------------------------------------
def veg_lag_correlation(vegetation_df: pd.DataFrame, climate_df: pd.DataFrame,
                        lags: range, rainy_season_onset: pd.DataFrame, months_in_season: list,
                        veg_var_name: str, clim_var_name: str) -> pd.DataFrame:
    """ 
    This function calculates the correlation between vegetation and climate datasets for a range of lag (time shift) values.
    The function filters the data for the rainy season months and calculates the correlation between the vegetation and climate datasets for each lag value.

    Parameters:
    vegetation_df (pd.DataFrame): A DataFrame containing the vegetation response dataset (e.g., GPP, ET, NDVI, etc.).
    climate_df (pd.DataFrame): A DataFrame containing the climate dataset (e.g., soil moisture, rainfall, temperature, etc.).
    lags (range): A range of lag values to test the correlation between the vegetation and climate datasets.
    rainy_season_onset (pd.DataFrame): A DataFrame containing the onset date of the rainy season for each year.
    months_in_season (list): A list of integers representing the months of the rainy season.
    veg_var_name (str): The column name in the vegetation DataFrame to be used for correlation.
    clim_var_name (str): The column name in the climate DataFrame to be used for correlation.

    Returns:
    pd.DataFrame: A DataFrame containing the correlation values and statistics across different lags for each year.
    """

    # Initialize lists to store correlation values and p-values per year
    seasonal_correlation = {}
    seasonal_p_values = {}

    # Loop through each year in the dataset
    for year, onset_date in zip(vegetation_df.index.year.unique(), rainy_season_onset['date']):
        #-----------------------------------------------------------------------------------------------
        # Filter the data for the rainy season of the current year
        veg_df_season = vegetation_df[(vegetation_df.index.year == year) & (vegetation_df.index.month.isin(months_in_season))]
        veg_df_season = veg_df_season[veg_df_season.index >= onset_date]

        clim_df_season = climate_df[(climate_df.index.year == year) & (climate_df.index.month.isin(months_in_season))]
        clim_df_season = clim_df_season[clim_df_season.index >= onset_date]

        #-----------------------------------------------------------------------------------------------
       
        # Initialize lists to store correlation values and p-values for the current season
        correlation_values = []
        p_values = []

        for lag in lags:
            # Shift vegetation response dataset backward by the lag value
            veg_df_lagged = veg_df_season.shift(lag).dropna()

            # Align both datasets to cover the same period
            ts_clim_df_lagged = clim_df_season.loc[veg_df_lagged.index]

            # Calculate correlation and p-value
            if not veg_df_lagged.empty and not ts_clim_df_lagged.empty:
                corr, p_value = pearsonr(veg_df_lagged[veg_var_name].values.flatten(), ts_clim_df_lagged[clim_var_name].values.flatten())
                correlation_values.append(corr)
                p_values.append(p_value)

        # Store the correlation and p-values for the current year
        seasonal_correlation[year] = correlation_values
        seasonal_p_values[year] = p_values

    # Create a DataFrame to store correlations for each year
    seasonal_correlation_df = pd.DataFrame(seasonal_correlation, index=lags)

    # Calculate statistics across all years
    max_correlations = seasonal_correlation_df.max(axis=1)
    min_correlations = seasonal_correlation_df.min(axis=1)
    correlation_mean = seasonal_correlation_df.mean(axis=1)
    corr_std = seasonal_correlation_df.std(axis=1)

    # Concatenate the dataframes
    correlation_stats_df = pd.concat([max_correlations, min_correlations, correlation_mean, corr_std], axis=1)
    correlation_stats_df.columns = ['max_corr', 'min_corr', 'mean_corr', 'std_dev_corr']

    combined_df = pd.concat([seasonal_correlation_df, correlation_stats_df], axis=1)

    return combined_df

    