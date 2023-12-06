import copy
import datetime
import os
import platform
import sys
import tempfile

# Suppress most (but not all) warnings
import warnings
from glob import glob

#from boto3.s3.connection import S3Connection
import boto3
import cartopy.crs as ccrs
import netCDF4
import numpy as np
import pyart
from matplotlib import pyplot as plt

warnings.simplefilter('ignore') 

import boto3
import botocore
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import fsspec
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapefile as shp
import tqdm
from botocore.handlers import disable_signing
from metpy.cbook import get_test_data
from metpy.plots import USCOUNTIES

# import nexradaws
# import pytz




# Magic command that forces all graphical output to appear in this notebook.
# %matplotlib inline






def _nearestDate(dates, pivot):
    return min(dates, key=lambda x: abs(x - pivot))




# "KMUX" "march 12 - 15 2023" "selinas california" "atmospheric river, dumped a bunch of water, flooding", "poverty ridden area" 

def get_radar_from_aws(site, datetime_t):
    """
    Get the closest volume of NEXRAD data to a particular datetime.
    Parameters
    ----------
    site : string
        four letter radar designation
    datetime_t : datetime
        desired date time
    Returns
    -------
    radar : Py-ART Radar Object
        Radar closest to the queried datetime
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/migrations3.html
    """

    # First create the query string for the bucket knowing
    # how NOAA and AWS store the data
    my_pref = datetime_t.strftime('%Y/%m/%d/') + site

    # Connect to the bucket
    
    s3_resource = boto3.resource('s3')
    s3_resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)


    # we are going to create a list of keys and datetimes to allow easy searching
    keys = []
    datetimes = []

    
    # populate the list
    buc = s3_resource.Bucket("noaa-nexrad-level2")
    for bucket in buc.objects.filter(Prefix=my_pref):
        this_str = str(bucket.key)
        #print(f" {this_str}")
        if 'gz' in this_str:
            endme = this_str[-22:-4]
            fmt = '%Y%m%d_%H%M%S_V0'
            dt = datetime_t.strptime(endme, fmt)
            datetimes.append(dt)
            keys.append(bucket)

        if this_str[-3::] == 'V06':
            endme = this_str[-19::]
            fmt = '%Y%m%d_%H%M%S_V06'
            dt = datetime_t.strptime(endme, fmt)
            datetimes.append(dt)
            keys.append(bucket)

    # find the closest available radar to your datetime
    closest_datetime = _nearestDate(datetimes, datetime_t)
    index = datetimes.index(closest_datetime)
    
    localfile = "./bucket_out.out"
    with open(localfile , 'wb') as data:
        obj = keys[index]
        obj.Object().download_fileobj(data)
    
    # print(keys[index])
    # .download_fileobj( keys[index], localfile)
    # keys[index].get_contents_to_filename(localfile.name)
    radar = pyart.io.read(localfile)
    return radar





def plot_radar_image(radar, radar_start_date):


    ymd_string = datetime.datetime.strftime(radar_start_date, '%Y%m%d')
    hms_string = datetime.datetime.strftime(radar_start_date, '%H%M%S')
    lats = radar.gate_latitude
    lons = radar.gate_longitude
    height = radar.gate_altitude
    lats = radar.gate_latitude
    lons = radar.gate_longitude
    min_lon = lons['data'].min()
    min_lat = lats['data'].min()
    max_lat = lats['data'].max()
    max_lon = lons['data'].max()



        

    
    lat_lines = np.arange(min_lat, max_lat, 1)
    lon_lines = np.arange(min_lon, max_lon, .1)



    
    # Setting projection and ploting the lowest tilt
    projection = ccrs.LambertConformal(central_latitude=radar.latitude['data'][0],
                                       central_longitude=radar.longitude['data'][0])
    
    
    fig = plt.figure(figsize=(12, 10), dpi=150)
    fig.set_facecolor('white')
    
    # create axis using cartopy projections, add in coastlines & grid
    ax = fig.add_subplot(1,1,1,projection=projection )
    
    
    # tmp = ax.axis('off') # Turn off enclosing frame
    display = pyart.graph.RadarMapDisplay(radar)
    
    #ax.add_feature(cfeature.ShapelyFeaturec(income_counties, projection=projection )  )
    
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.add_feature(cfeature.STATES, linewidth=3)
    # ax.add_feature(USCOUNTIES, alpha=0.4)
    ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.10, edgecolor='black')



    
    # Setting projection and ploting the lowest tilt
    projection = ccrs.LambertConformal(central_latitude=radar.latitude['data'][0],
                                       central_longitude=radar.longitude['data'][0])
    # Built-in Py-ART function to plot PPIs.
    
    
    # Latitude and longitude of the marker
    marker_lat = 29.7604  # Replace with your desired latitude
    marker_lon = -95.3698  # Replace with your desired longitude
    
    
    # Add a star marker at the specified latitude and longitude
    # ax.plot(marker_lon, marker_lat, marker='*', markersize=30, color='k', zorder=22)


    
    
    # See: 
    display.plot_ppi_map('reflectivity', sweep = 0, vmin = 0, vmax = 46,
                         ax=ax,
                         min_lon = marker_lon-4, max_lon = max_lon+2.5,
                         min_lat = marker_lat-4, max_lat = max_lat+2,
                         resolution = '10m', projection = projection, fig = fig, cmap = "pyart_HomeyerRainbow")

    display.plot_point(marker_lon, marker_lat, symbol = "k*", markersize=12)
    output_path = f"./{station}_3"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    plt.savefig(   f"{output_path}/ref_"+ station + "_"  + ymd_string + hms_string + ".png" )

    return plt






#station = 'KHGX'
#station = 'KLCH'
# station = "KMUX" 

# reader = shpreader.Reader('./texas_state.shp')
print(f" current_working_directory: {os.getcwd()} ")
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)





station = 'KHGX'
start_date = datetime.datetime(2017, 8, 25, 1, 00)
end_date = datetime.datetime(2017, 8, 31, 11, 00)




delta = datetime.timedelta(hours=1)
current_date = start_date
while tqdm.tqdm(current_date <= end_date):
    current_date += delta
    print(f"{current_date}")
    radar = get_radar_from_aws(station, current_date)
    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'], only_use_cftime_datetimes = False, only_use_python_datetimes = True)
    out = plot_radar_image(radar, radar_start_date)



