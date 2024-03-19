#!/usr/bin/env python
import cdsapi
from datetime import datetime, timedelta
import cdstoolbox as ct
import logging, sys, os
import yaml

with open('/scratch/gilbreth/gupt1075/FourCastNet_gil/.cdsapirc', 'r') as f:
        credentials = yaml.safe_load(f)

c = cdsapi.Client(url=credentials['url'], key=credentials['key'])
start_time = datetime(2020, 1, 1, 0, 0, 0)
end_time = datetime(2021, 1, 1, 0, 0, 0)
# timedelta = timedelta(hours=6)





@ct.application(title='Retrieve Data')
@ct.output.download()
def retrieve_sample_data():
    """
    Application main steps:

    - retrieve a variable from CDS Catalogue
    - produce a link to download it.
    """

    data = ct.catalogue.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': '2m_temperature',
            'product_type': 'reanalysis',
            'year': '2017',
            'month': '01',
            'day': '02',
            'time': '12:00',
        }
    )
    return data



c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'date'    :  f"{start_time.strftime('%Y-%m-%d')}/to/{end_time.strftime('%Y-%m-%d')}",
        'pressure_level': [ '50', '500', '850','1000',],
        'product_type': 'reanalysis',
        'variable': [
            'geopotential', 'relative_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind',
        ],

        'time': [ '00:00', '06:00', '12:00','18:00',], 
        'area'    : 'global', 
        'format': 'netcdf',    
            
    },
    '/scratch/gilbreth/gupt1075/copernicus_data/2020_pl.nc')





c.retrieve('reanalysis-era5-complete', {
    # Removed obsolete comments and streamlined the explanation
    'date'    :  f"{start_time.strftime('%Y-%m-%d')}/to/{end_time.strftime('%Y-%m-%d')}",  # Specify the range of dates to download
    'levelist': '500',              # pressure Levels 
    'levtype' : 'pl',                        # Model levels
    'param'   : '137',                       # Parameter code for geopotential
    'stream'  : 'oper',                      # Operational data stream
    'time'    : '00/06/12/18',               # Times of day (in shorthand notation)
    'type'    : 'an',                        # Analysis type
    'area'    : 'global',              # Geographical subarea: North/West/South/East
    'grid'    : '0.25/0.25',                   # Grid resolution: latitude/longitude
    'format'  : 'netcdf',                    # Output format, requires 'grid' to be specified
}, f"{output_path}" + f"tcwv_{start_time.strftime('%Y-%m-%d')}_to_{end_time.strftime('%Y-%m-%d')}_" + "ERA5-pl-z500.25.nc")        # Output filename






# c.retrieve('reanalysis-era5-complete', {
#     # Removed obsolete comments and streamlined the explanation
#     'date'    :  f"{start_time.strftime('%Y-%m-%d')}/to/{end_time.strftime('%Y-%m-%d')}",  # Specify the range of dates to download
#     'levelist': '500',              # pressure Levels 
#     'levtype' : 'pl',                        # Model levels
#     'param'   : '129',                       # Parameter code for geopotential
#     'stream'  : 'oper',                      # Operational data stream
#     'time'    : '00/06/12/18',               # Times of day (in shorthand notation)
#     'type'    : 'an',                        # Analysis type
#     'area'    : 'global',              # Geographical subarea: North/West/South/East
#     'grid'    : '0.25/0.25',                   # Grid resolution: latitude/longitude
#     'format'  : 'netcdf',                    # Output format, requires 'grid' to be specified
# }, f"{output_path}" + f"NETCDF_{start_time.strftime('%Y-%m-%d')}_to_{end_time.strftime('%Y-%m-%d')}_" + "ERA5-pl-z500.25.nc")        # Output filename
