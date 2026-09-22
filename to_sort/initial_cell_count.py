#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import skimage as ski

# load the image
image = ski.data.human_mitosis()

# set the number of classes for cell detection
# thresholds_2classes = ski.filters.threshold_multiotsu(image, classes=2)
thresholds = ski.filters.threshold_multiotsu(image, classes=3)


# Just set a single threshold?
threshold = ski.filters.threshold_otsu(image)

# get all the cells, which would be above the threshold
cells = image > threshold

# distance transform the centroids
distance = ndi.distance_transform_edt(cells)

# find local maxima (markers)
local_max_coords = ski.feature.peak_local_max(distance, min_distance=10)
local_max_mask = np.zeros(distance.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
markers = ndi.label(local_max_mask)[0]

# 5. Watershed
segmented_cells = ski.segmentation.watershed(-distance, markers, mask=cells)

# # plot the impact
regions = np.digitize(image, bins = thresholds)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title('Original')
ax[0].set_axis_off()
ax[1].imshow(regions)
ax[1].set_title('Multi-Otsu thresholding')
ax[1].set_axis_off()
plt.show()
#
cells = image > thresholds[0]
# dividing = image > thresholds[1]
labeled_cells = ski.measure.label(cells)

final_cells, final_cell_num  = ski.measure.label(labeled_cells > 0, return_num = True)

# fig, ax = plt.subplots(ncols=2, figsize=(10, 10))

plt.figure(figsize=(10, 10))
plt.imshow(ski.color.label2rgb(final_cells, bg_label=0), cmap='nipy_spectral')
plt.title('Labeled Cells')
plt.axis('off')

for region in ski.measure.regionprops(final_cells):
    # Take the centroid of the region and use it for placing the label
    y, x = region.centroid
    plt.text(x, y+30, f"Cell: {region.label}", color='white', fontsize=12, ha='center', va='center')

# plt.show()
