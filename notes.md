## For CCRS projection depending upon the use case

# polars: sterographic projection,
# tropics : mercator
# 30 N and 30 South is called as Tropics
# continnent us: lambert conformal



# plot  for the following days z500 and compare it to official era5 cds calls from ecmwf

# how deep 100 * 4 frames (100 days) 2018 january and 2018 febraury , 2019 januarry 
# run code for 2017 december 1st and 2018 march 1st  , 2018 june 1st, 2018 september
# robinson preserves the original distance in highger latitued much better than mercators

# precipitaiton modeling using sfno




#### February 3

# the plots for z500 are of units m2/sec and it around 5 km from surface where we get the z500 values : the range of the values are of order 50,000
# the code doesn't seem to be normalizing the original netcdf files, whilst there is explicit normalization whiel plotting
# try to load the grib file and plot the z500 in the same nortpolarsterographic or orthographic projection iwth the ensemble outoput





# 9th february
# get the correct acc values, also get the correct climatorlogy and play around with differnt climatology, log out all the variables in acc caclclation

<!-- get robinson plots for each of th e4 esnembles for one days -->