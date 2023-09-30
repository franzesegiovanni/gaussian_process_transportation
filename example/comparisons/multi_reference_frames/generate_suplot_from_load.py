import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

script_path = os.path.dirname(__file__) + '/figs/'

# Load the plots from the image files
plots = [
    imread(script_path + 'hmm.png'),
    imread(script_path + 'tp_gmm.png'),
    imread(script_path + 'dmp.png'),
    imread(script_path + 'gpt.png'),
    imread(script_path + 'hmm_new.png'),
    imread(script_path + 'tp_gmm_new.png'),
    imread(script_path + 'dmp_new.png'),
    imread(script_path + 'gpt_new.png')
]

# Calculate the aspect ratio of the source images
aspect_ratios = [plot.shape[1] / plot.shape[0] for plot in plots]

# Calculate the number of rows and columns in the subplot grid
num_cols = 4
num_rows = -(-len(plots) // num_cols)  # Round up to the nearest integer

# Calculate the size of the subplots based on the aspect ratios
subplot_width = 16 / num_cols
subplot_height = subplot_width / min(aspect_ratios)

# Create a subplot with the plots arranged in a grid
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, subplot_height * num_rows))
plt.subplots_adjust(wspace=0, hspace=0.1)

# Plot each image on a separate subplot
for i, ax in enumerate(axes.flat):
    if i < len(plots):
        ax.imshow(plots[i])
        ax.axis('off')

fig.tight_layout()

# Save the figure
fig.savefig(script_path + 'comparison.png', dpi=1200, bbox_inches='tight')
plt.show()