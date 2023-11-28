import datetime
import os

import dotenv
import xarray

dotenv.load_dotenv()

from earth2mip import inference_ensemble, registry
from earth2mip.initial_conditions import cds

import earth2mip.networks.dlwp as dlwp
import earth2mip.networks.pangu as pangu



cds_api = os.path.join(os.path.expanduser("~"), ".cdsapirc")
if not os.path.exists(cds_api):
    uid = input("Enter in CDS UID (e.g. 123456): ")
    key = input("Enter your CDS API key (e.g. 12345678-1234-1234-1234-123456123456): ")
    # Write to config file for CDS library
    with open(cds_api, "w") as f:
        f.write("url: https://cds.climate.copernicus.eu/api/v2\n")
        f.write(f"key: {uid}:{key}\n")
        
        
        
        
time = datetime.datetime(2018, 1, 1)

# DLWP datasource
dlwp_data_source = cds.DataSource(dlwp_inference_model.in_channel_names)

# Pangu datasource, this is much simplier since pangu only uses one timestep as an input
pangu_data_source = cds.DataSource(pangu_inference_model.in_channel_names)



package = PrecipAFNO.load_package()

model = PrecipAFNO.load_diagnostic(package)

x = torch.randn(1, 4, 720, 1440)

out = model(x)

out.shape
(1, 1, 721, 1440)



import matplotlib.pyplot as plt

# Open dataset from saved NetCDFs
pangu_ds = xarray.open_dataarray(f"{output_dir}/pangu_inference_out.nc")
dlwp_ds = xarray.open_dataarray(f"{output_dir}/dlwp_inference_out.nc")

# Get data-arrays at 12 hour steps
pangu_arr = pangu_ds.sel(channel="z500").values[::2]
dlwp_arr = dlwp_ds.sel(channel="z500").values
# Plot
plt.close("all")
fig, axs = plt.subplots(2, 13, figsize=(13 * 4, 5))
for i in range(13):
    axs[0, i].imshow(dlwp_arr[i, 0])
    axs[1, i].imshow(pangu_arr[i, 0])
    axs[0, i].set_title(time + datetime.timedelta(hours=12 * i))

axs[0, 0].set_ylabel("DLWP")
axs[1, 0].set_ylabel("Pangu")
plt.suptitle("z500 DLWP vs Pangu")
plt.savefig(f"{output_dir}/pangu_dlwp_z500.png")
        
        

# Output directoy
output_dir = "outputs/02_model_comparison"
os.makedirs(output_dir, exist_ok=True)

print("Loading models into memory")
# Load DLWP model from registry
package = registry.get_model("dlwp")
dlwp_inference_model = dlwp.load(package)

# Load Pangu model(s) from registry
package = registry.get_model("pangu")
pangu_inference_model = pangu.load(package)