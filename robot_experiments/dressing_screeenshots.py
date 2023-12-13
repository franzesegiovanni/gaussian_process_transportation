import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import os
directory = os.path.dirname(__file__)

print(directory)
# Set the path to the folder containing the screenshots
folder_path =directory + "/video_frames/dressing/"

# Get a list of all the files in the folder
files = os.listdir(folder_path)

# Filter out any non-image files
image_files = [f for f in files if f.endswith(".png")]

# Sort the image files alphabetically
image_files.sort()

# Create a figure with 2 rows and 5 columns
fig, axs = plt.subplots(2, 5, constrained_layout=True, figsize=(9.5, 2.8))

# Loop through the image files and plot them in the subplots
for i, image_file in enumerate(image_files):
    # Calculate the row and column indices for this subplot
    row = i // 5
    col = i % 5

    # Load the image and plot it in the appropriate subplot
    image_path = os.path.join(folder_path, image_file)
    image=Image.open(image_path)
    # Crop the image
    left = 600
    top = 150
    right = image.width
    bottom = image.height
    image = image.crop((left, top, right, bottom))
    enhancer = ImageEnhance.Brightness(image)
    brightness_factor = 1.3  # Adjust the value as needed (1.0 means no change)
    image_brightened = enhancer.enhance(brightness_factor)
    axs[row, col].imshow(image_brightened)
    axs[row, col].axis("off")
    if i==0:

        axs[row, col].text(0.07, 0.85, 'D', horizontalalignment='center', verticalalignment='center', transform=axs[row, col].transAxes, fontsize=20)
    else:
        axs[row, col].text(0.07, 0.85, str(i), horizontalalignment='center', verticalalignment='center', transform=axs[row, col].transAxes, fontsize=20)    

plt.subplots_adjust(hspace=0, wspace=-0.17) # Set the vertical space between rows to 0

# plt.tight_layout()

plt.savefig(directory+'/figures/video_dressing.pdf', bbox_inches='tight', dpi=600)
# Show the plot
plt.show()
