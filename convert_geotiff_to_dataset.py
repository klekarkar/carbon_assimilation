import os
import xarray as xr
import numpy as np
import pandas as pd
from osgeo import gdal
from datetime import datetime, timedelta

def merge_Gtiffs_to_Dataset(files: list,variable: str, units: str):
    """
    script to convert a list of geotiff files to a single xarray dataset
    :param files: list of geotiff files
    :param variable: variable name
    :param year: year of the data
    :param month: month of the data
    :param day: day of the data
    :param units: units of the data
    :return: xarray dataset
    """
    array_list=[]
    files.sort()
    for file in files:
        
        #convert year and DOY to datetime
        # Combine extracted strings into a single date string and convert to datetime
        #for GLDAS. FLDAS files
        # year = os.path.basename(file).split('_')[1][:4]
        # month = os.path.basename(file).split('_')[1][4:6]
        # day = os.path.basename(file).split('_')[1][6:8]
        #...........................................................................
        #for modis LST files
        # year = os.path.basename(file).split('_')[1]
        # month = os.path.basename(file).split('_')[2]
        # day = os.path.basename(file).split('_')[3]
        #...........................................................................
        #for plm_v2 files
        # year=os.path.basename(file).split('_')[1]
        # month=os.path.basename(file).split('_')[2]
        # day=os.path.basename(file).split('_')[3]
        #...........................................................................
        
        #for files with jday
        year = os.path.basename(file).split('.')[1][1:5]
        julian_day = os.path.basename(file).split('.')[1][5:]

        # Convert Julian day to a normal date
        date_string = f"{year}{julian_day}"
        date = pd.to_datetime(date_string, format='%Y%j')

        #...........................................................................

        # date_string = f"{year}{month}{day}"
        # date = pd.to_datetime(date_string, format='%Y%m%d')

        # date_string = f"{year}{month}{day}"
        # date = pd.to_datetime(date_string, format='%Y%m%d')

        #read the first file to get the dimensions
        with gdal.Open(file) as ds:
            band = ds.GetRasterBand(1)
            arr = band.ReadAsArray()
            nlat,nlon = np.shape(arr)
            b = ds.GetGeoTransform() #bbox, interval

            #get the number of rows and columns and multiply by the interval, then add to the origin to get the coordinates
            lon = np.arange(nlon)*b[1]+b[0]
            lat = np.arange(nlat)*b[5]+b[3]

            #assign the coordinates to the array
            arr = xr.DataArray(arr,coords=[lat,lon],dims=['lat','lon'])

            #assign the date to the array
            arr = arr.expand_dims('time')
            arr['time'] = [date]

            #assign nodata value
            arr = arr.where(arr!=band.GetNoDataValue())

            #to reduce file size, convert to float32
            arr=arr.astype('float32')

            #set projection
            arr.attrs['crs'] = 'EPSG:4326'

            #assign variable properties
            arr.attrs['units'] = units
            arr.attrs['long_name'] = f'{variable}'
            arr.attrs['source'] = 'NASA GES DISC at NASA Goddard Earth Sciences Data and Information Services Center'
            arr.attrs['scale_factor'] = 1

            array_list.append(arr)
            print(f"Processed {file}", end='\r')

    #concatenate the list of arrays into a single xarray dataset
    print(f"Saving the dataset \n")
    npp_xr=xr.concat(array_list,dim='time')
    npp_xr.name = variable

    print(f"Saved the {variable} dataset")

    return npp_xr

    #save the dataset to a netcdf file
    # npp_xr.to_netcdf(output_file)

    # print(f"NetCDF file saved to {output_file}")