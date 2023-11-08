import os
import matplotlib.pyplot as plt
from PIL import Image
import os
directory = os.path.dirname(__file__)

print(directory)
# Set the path to the folder containing the screenshots
folder_path =directory + "/video_frames/reshelving/place/"

# Get a list of all the files in the folder
files = os.listdir(folder_path)

# Filter out any non-image files
image_files = [f for f in files if f.endswith(".png")]

# Sort the image files alphabetically
image_files.sort()
plt.figure(figsize=(12, 3))
for i, image_file in enumerate(image_files):
    # Calculate the row and column indices for this subplot

    # Load the image and plot it in the appropriate subplot
    image_path = os.path.join(folder_path, image_file)
    image=Image.open(image_path)
    # Crop the image
    left = 200
    top = 0
    right = image.width
    bottom = image.height
    image = image.crop((left, top, right, bottom))
    plt.imshow(image, alpha=1-0.2*i)
    plt.axis('off')
plt.show()
    
