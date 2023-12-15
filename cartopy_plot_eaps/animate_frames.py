import glob

from PIL import Image

# Create the frames
frames = []
imgs = sorted(glob.glob("./KHGX_3/ref_KHGX_2017*.png"))
for i in imgs:
    print(f" currently working on {i}  ")
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save(
    "./hurricane_harvey_hourly_60x.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=100,
    loop=0,
)