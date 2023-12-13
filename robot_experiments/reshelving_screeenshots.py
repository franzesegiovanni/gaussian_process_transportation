import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import os
directory = os.path.dirname(__file__)

print(directory)
# Set the path to the folder containing the screenshots
folder_path =directory + "/video_frames/reshelving/new_plot/"

# Get a list of all the files in the folder
files = os.listdir(folder_path)

# Filter out any non-image files
image_files = [f for f in files if f.endswith(".png")]

# Sort the image files alphabetically
image_files.sort()

# Create a figure with 2 rows and 5 columns
fig, axs = plt.subplots(2, 6, constrained_layout=True, figsize=(9.5, 2.8))

# Loop through the image files and plot them in the subplots
for i, image_file in enumerate(image_files):
    # Calculate the row and column indices for this subplot
    row = i % 2
    col = i // 2

    # Load the image and plot it in the appropriate subplot
    image_path = os.path.join(folder_path, image_file)
    image=Image.open(image_path)
    # Crop the image
    left = 300
    top = 0
    right = image.width-400
    bottom = image.height
    image = image.crop((left, top, right, bottom))
    # Reduce yellow tint in the image
    enhancer_color = ImageEnhance.Color(image)
    color_factor = 1.5  # Adjust the value as needed (1.0 means no change)

    image = enhancer_color.enhance(color_factor)

    # Increase brightness
    enhancer_brightness = ImageEnhance.Brightness(image)
    brightness_factor = 1.1  # Adjust the value as needed (1.0 means no change)

    image = enhancer_brightness.enhance(brightness_factor)
    axs[row, col].imshow(image)
    axs[row, col].axis("off")
    if i<2:

        axs[row, col].text(0.07, 0.85, 'D', horizontalalignment='center', verticalalignment='center', transform=axs[row, col].transAxes, fontsize=20)
    else:
        axs[row, col].text(0.07, 0.85, str(i //2), horizontalalignment='center', verticalalignment='center', transform=axs[row, col].transAxes, fontsize=20)   
        # Add text to the left of the rows
    if col == 0 and row==0:  # only add text for the first column
        axs[row, col].text(-0.1, 0.5, 'Place', horizontalalignment='center', verticalalignment='center', transform=axs[row, col].transAxes, fontsize=15, rotation=90)
    if col == 0 and row==1:  # only add text for the first column
        axs[row, col].text(-0.1, 0.5, 'Pick', horizontalalignment='center', verticalalignment='center', transform=axs[row, col].transAxes, fontsize=15, rotation=90)
    



plt.subplots_adjust(hspace=0, wspace=0) # Set the vertical space between rows to 0

plt.tight_layout()

plt.savefig(directory+'/figures/video_reshalving.pdf', bbox_inches='tight', dpi=600)
# Show the plot
plt.show()
