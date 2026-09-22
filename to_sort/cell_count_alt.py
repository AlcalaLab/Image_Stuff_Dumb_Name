#!/usr/bin/env python3

"""
Classic approaches to segmenting cell images. This does NOT use ML/DL approaches,
but instead relies on classic thresholding approaches.

This is a TESTING ground for manipulating images and parsing them...

Dependencies:
- skimage
- scipy
"""

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy import ndimage as ndi

import skimage as ski



def prepare_img_file(
        image_file: str,
        outdir: str,
        make_plots: bool = True,
        brightfield: bool = True,
        verbose: bool = True
        ) -> arr:
    """
    Prepares the images (tif or png).

    Parameters
    ----------
    image_file:    path to the image to process
    outdir:        output directory to store output data
    make_plots:    include plots of intermediate data/steps
    brightfield:   is the image from brightfield
    verbose:       be loud and proud


    Returns
    ----------
    grayscale_image: nd array of the processed image
    """

    if make_plots:
        Path(f'{outdir}/Intermediate_Plots/').mkdir(exist_ok = True, parents = True)
        Path(f'{outdir}/Final_Plots/').mkdir(exist_ok = True, parents = True)

    # parse the initial image
    initial_image = ski.io.imread(img_file)
    image_name = img_file.rpartition('/').rpartition(".")[0]

    # files from Maisie tend to have (R, G, B, and alpha), so convert to gray,
    # and exlude the alpha layer.
    gray_image = ski.color.rgb2gray(initial_image[:,:,:3])
    grayscale_image =  ski.util.invert(ski.util.img_as_ubyte(gray_image))

    # contour plot of the grayscaled imaget image
    # useful to get an idea of the qualities in the image
    if make_plots:
        fig, ax = plt.subplots(figsize=(5, 5))

        # plot the contours
        qcs = ax.contour(grayscale_image, origin = image_name)

        ax.set_title(f'Contour plot of {image_name}')

        plt.tight_layout()

        plt.savefig(f'{outdir}/Intermediate_Plots/{image_name}.Contor_Plot.png')

        plt.clf()
        # plot the histogram of gray values in the image
        hist, hist_centers = ski.exposure.histogram(grayscale_image)

        fig, axes = plt.subplots(1, 2, figsize = (8, 3))
        axes[0].imshow(grayscale_image, cmap = plt.cm.gray)
        axes[0].set_axis_off()
        axes[0].set_title('Inverted Initial Grayscale Image')

        axes[1].plot(hist_centers, hist, lw = 2)
        axes[1].set_title('Histogram of Gray Values')

        plt.tight_layout()

        plt.savefig(f'{outdir}/Intermediate_Plots/{image_name}.Contor_Plot.png')
        plt.clf()

    return grayscale_image


# #--- edge-based segmentation ---#
# """
# Initial thoughts:
# Pretty simple and works reasonably well for segmenting cells from
# Tetrahymena pyriformis cultures.
#
# Might want to concentrate cells (0.2 mL), resuspend in a smaller volume (40uL)
# for imaging...
# """
# # Delineate edges using Canny edge detector
# edges = ski.feature.canny(grayscale_image)
#
# fig, ax = plt.subplots(figsize = (4, 3))
# ax.imshow(edges, cmap = plt.cm.gray)
# ax.set_title('Canny detector')
# ax.set_axis_off()
# plt.show()
#
# # Fill in the edges with mathematical morphology
# fill_cells = ndi.binary_fill_holes(edges)
#
# fig, ax = plt.subplots(figsize = (4, 3))
# ax.imshow(fill_cells, cmap = plt.cm.gray)
# ax.set_title('filling the holes')
# ax.set_axis_off()
# plt.show()
#
# # Remove spurious fills
# cells_cleaned = ski.morphology.remove_small_objects(fill_cells, min_size = 150)
#
# fig, ax = plt.subplots(figsize = (4, 3))
# ax.imshow(cells_cleaned, cmap=plt.cm.gray)
# ax.set_title('removing small objects')
# ax.set_axis_off()
# plt.show()

#--- region-based segmentation ---#
"""
Initial thoughts:
This is the way to go! I love it!!!

Might want to concentrate cells (0.2 mL), resuspend in a smaller volume (40uL)
for imaging...
"""
# make an elevation map using the Sobel gradient
elevation_map = ski.filters.sobel(grayscale_image)

# plot the evelation map
fig, ax = plt.subplots(figsize = (4, 3))
ax.imshow(elevation_map, cmap = plt.cm.gray)
ax.set_title('elevation map')
ax.set_axis_off()
plt.show()


# Find markers of the background and cells based on the extremes in the histogram of
# gray scale values
markers = np.zeros_like(grayscale_image)
markers[grayscale_image < 10] = 1
markers[grayscale_image > 100] = 2

fig, ax = plt.subplots(figsize = (4, 3))
ax.imshow(markers, cmap = plt.cm.nipy_spectral)
ax.set_title('markers')
ax.set_axis_off()
plt.show()

# Use watershem segmentation to fill regions of the elevation map
segmented_cells = ski.segmentation.watershed(elevation_map, markers)

fig, ax = plt.subplots(figsize = (4, 3))
# ax.imshow(segmented_cells, cmap = plt.cm.gray)
# ax.set_title('segmentation')
# ax.set_axis_off()
# plt.show()

# # Remove spurious fills
# cells_cleaned = ski.morphology.remove_small_objects(segmented_cells, min_size = 200)
#
# fig, ax = plt.subplots(figsize = (4, 3))
# ax.imshow(cells_cleaned, cmap = plt.cm.gray)
# ax.set_title('removing small objects')
# ax.set_axis_off()
# plt.show()

# Fill in the segmented cells using mathematical morphology
filled_segmented_cells = ndi.binary_fill_holes(segmented_cells - 1)

# Remove spurious fills
cells_cleaned = ski.morphology.remove_small_objects(filled_segmented_cells, min_size = 200)
#
# fig, ax = plt.subplots(figsize = (4, 3))
# ax.imshow(cells_cleaned, cmap = plt.cm.gray)
# ax.set_title('removing small objects')
# ax.set_axis_off()
# plt.show()

# Label the cells
labeled_cells, _ = ndi.label(cells_cleaned)
image_label_overlay = ski.color.label2rgb(labeled_cells, image = initial_image[:,:,:3], bg_label=0)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(initial_image[:,:,:3], cmap = plt.cm.gray)
axes[0].contour(cells_cleaned, [0.5], linewidths = 1.2, colors='y')
axes[1].imshow(image_label_overlay)

for a in axes:
    a.set_axis_off()

fig.tight_layout()

plt.show()












# set the number of classes for cell detection
# thresholds_2classes = ski.filters.threshold_multiotsu(image, classes=2)
thresholds = ski.filters.threshold_multiotsu(grayscale_image, classes = 3)

# Just set a single threshold?
# threshold = ski.filters.threshold_otsu(grayscale_image)

# get all the cells, which would be above the threshold
cells = grayscale_image > thresholds[0]

# distance transform the centroids
distance = ndi.distance_transform_edt(cells)

# find local maxima (markers)
local_max_coords = ski.feature.peak_local_max(distance, min_distance=10)
local_max_mask = np.zeros(distance.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
markers = ndi.label(local_max_mask)[0]

# 5. Watershed
segmented_cells = ski.segmentation.watershed(-distance, markers, mask = cells)

# # plot the impact
regions = np.digitize(grayscale_image, bins = thresholds)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(grayscale_image)
ax[0].set_title('Original')
ax[0].set_axis_off()
ax[1].imshow(regions)
ax[1].set_title('Multi-Otsu thresholding')
ax[1].set_axis_off()
plt.show()


cells = grayscale_image > thresholds[1]
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
