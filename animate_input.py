import h5py
import numpy as np
import sys, os


import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import logging

# from matplotlib import animation
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.dpi'] = 400  
plt.ioff()
plt.rcParams['animation.ffmpeg_path'] = '/scratch/gilbreth/gupt1075/FourCastNet/'



path = "/depot/gdsp/data/gupt1075/fourcastnet/data/FCN_ERA5_data_v0/train/"
filename2 = "/scratch/gilbreth/gupt1075/ERA5_expts_gtc/autoregressive_predictions_z500.h5"
filename = os.path.join(path,"1979.h5")


list1 = []
with h5py.File(filename2, "r") as hf:
    logging.warning( f" {hf.keys()} " )
    ndarray = np.array(hf["fields"][:1400, 0])
    list1.append(ndarray)
    
data = list1[0]
logging.warning(f"data_shape {data.shape}")



fig = plt.figure( figsize=(12,12) )

a = data[0]
im = plt.imshow(a, interpolation='none', aspect='auto')

def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )
    im.set_array(data[i])
    return [im]

anim = animation.FuncAnimation(fig,  animate_func,  frames = nSeconds * fps,interval = 1000 / fps, repeat=True  )
writergif = animation.PillowWriter(fps=30) 

# extra_args=['-vcodec', 'libx264']
anim.save( "input.gif" , writer=writergif)
# anim.save('input_gif.mp4', fps=fps )

logging.warning('Done!')








# with h5py.File(filename, "r") as f:
#     # Print all root level object names (aka keys) 
#     # these can be group or dataset names 
#     print(f"Keys: {f.keys()} ")
#     # get first object name/key; may or may NOT be a group
#     a_group_key = list(f.keys())[0]

#     # get the object type for a_group_key: usually group or dataset
#     print(type(f[a_group_key])) 

#     # If a_group_key is a group name, 
#     # this gets the object names in the group and returns as a list
#     data = list(f[a_group_key])

#     # If a_group_key is a dataset name, 
#     # this gets the dataset values and returns as a list
#     data = list(f[a_group_key])
#     # preferred methods to get dataset values:
#     ds_obj = f[a_group_key]      # returns as a h5py dataset object
#     ds_arr = f[a_group_key][()]  # returns as a numpy array
    
    
    

#     dset = f['key'][:]
#     img = Image.fromarray(dset.astype("uint8"), "RGB")
#     img.save("./test.png")
    
#     print(f" ds_obj: {ds_obj} ds_arr: {ds_arr} ")
    


fps = 30
nSeconds = 5

# hf = h5py.File(filename, 'r')
# ndarray = np.array(hf["fields"][:])
# print(ndarray.shape)
# First set up the figure, the axis, and the plot element we want to animate


# plt.show()  # Not required, it seems!
