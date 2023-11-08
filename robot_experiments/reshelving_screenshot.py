import os
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
directory = os.path.dirname(__file__)

print(directory)
# Set the path to the folder containing the screenshots
folder_path =directory + "/video_frames/reshelving/place/cropped/"

# Get a list of all the files in the folder
files = os.listdir(folder_path)

image_template = Image.open(folder_path + "template.png")
# Convert images to grayscale
left = 200
top = 0
right = image_template.width-1000
bottom = image_template.height-300
image_template = image_template.crop((left, top, right, bottom))
image_files = [f for f in files if f.endswith(".png") and not(f.startswith("template"))]

# Sort the image files alphabetically
image_files.sort()
plt.figure(figsize=(12, 3))
plt.imshow(image_template)
plt.axis('off')
for i, image_file in enumerate(image_files):
    # Calculate the row and column indices for this subplot

    # Load the image and plot it in the appropriate subplot
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path).convert("RGBA")
    image_array = np.copy(np.asarray(image))
    mask=image_array[:,:,0]==0
    image_array[:,:,-1][mask]=0
    image = Image.fromarray(image_array)
    image=image.crop((left, top, right, bottom))
    plt.imshow(image)
    plt.axis('off')
plt.savefig(directory+"/figures/reshelving_screenshot.pdf", bbox_inches='tight', pad_inches=0)    
plt.show()
    
