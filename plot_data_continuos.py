import numpy as np
from matplotlib import pyplot as plt
import random
from matplotlib.widgets import Button
# Load the MRI data
img = np.load("./data/training/images/image_100.npy")
img_mask = np.load("./data/training/masks/mask_100.npy")
# Select the T1n image
test_image_t1n = img[0, :, :, :]
img_mask = img_mask[0, :, :, :]
fig, ax = plt.subplots(figsize=(10, 8))
num_slices = test_image_t1n.shape[2]

# Display each slice
for n_slice in range(num_slices):
    # ax.imshow(test_image_t1n[:, :, n_slice],
    #          cmap="gray", interpolation='nearest')
    ax.imshow(img_mask[:, :, n_slice], cmap="Set1", alpha=0.1)
    ax.set_title(f'Image T1n, Slice {n_slice}')
    ax.axis('off')

    plt.draw()
    plt.pause(0.1)

    ax.cla()
