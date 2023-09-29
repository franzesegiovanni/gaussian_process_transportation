import matplotlib.pyplot as plt
from matplotlib.image import imread


# Load the first two plots from the image files
plot1 = imread('figs/hmm.png')
plot2 = imread('figs/transportation.png')
plot3 = imread('figs/hmm_new.png')
plot4 = imread('figs/transportation_new.png')

# Create a single subplot with the four plots in a row
plt.figure(figsize=(12, 4))  # Adjust the figure size as needed

# Plot 1
plt.subplot(2, 2, 1)
plt.imshow(plot1)
plt.axis('off')
plt.title('Plot 1')

# Plot 2
plt.subplot(2, 2, 2)
plt.imshow(plot2)
plt.axis('off')
plt.title('Plot 2')

# Plot 3
plt.subplot(2, 2, 3)
plt.imshow(plot3)
plt.axis('off')
plt.title('Plot 3')

# Plot 4
plt.subplot(2, 2, 4)
plt.imshow(plot4)
plt.axis('off')
plt.title('Plot 4')

plt.tight_layout()  # Adjust the spacing between subplots

# Display the combined subplot
plt.show()
