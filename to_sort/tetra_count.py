#!/usr/bin/env python3

"""
Classic approaches to segmenting cell images. This does NOT use ML/DL approaches,
but instead relies on classic thresholding approaches.

This is a TESTING ground for manipulating images and parsing them...

--------------------------------------------------------------------------------
Initial thoughts:
This is the way to go! I love it!!!

Might want to concentrate cells (0.2 mL), resuspend in a smaller volume (40uL)
for imaging...
--------------------------------------------------------------------------------

Dependencies:
- pandas
- skimage
- scipy
- tifffile
"""

import glob, sys, tifffile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from scipy import ndimage as ndi

import skimage as ski


def closest_to_val(
    hist,
    hist_centers) -> int:
    for n in range(len(hist)):
        if hist[n] >= max(hist) * 0.15:
            return int(hist_centers[n])


def prepare_img_file(
        image_file: str,
        outdir: str,
        make_plots: bool = True,
        brightfield: bool = True,
        verbose: bool = True
        ) -> list[arr | str]:
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
    initial_image = ski.io.imread(image_file)
    image_name = image_file.rpartition('/')[-1].rpartition(".")[0]
    image_name = image_name.rstrip('.ome.')

    # files from Maisie tend to have (R, G, B, and alpha), so convert to gray,
    # and exlude the alpha layer.
    gray_image = ski.color.rgb2gray(initial_image[:,:,:3])
    grayscale_image =  ski.util.invert(ski.util.img_as_ubyte(gray_image))

    # histogram of gray values in the image
    hist, hist_centers = ski.exposure.histogram(grayscale_image)
    try:
        min_hist_val = closest_to_val(hist, hist_centers)
    except:
        print(f'Error processing {img_file}')
        sys.exit(1)

    # contour plot of the grayscaled imaget image
    # useful to get an idea of the qualities in the image
    if make_plots:
        fig, ax = plt.subplots(figsize=(5, 5))

        # plot the contours
        qcs = ax.contour(grayscale_image, origin = 'image')
        ax.set_title(f'Contour plot of {image_name}', fontsize = 8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{outdir}/Intermediate_Plots/{image_name}.Contor_Plot.png', dpi = 300)
        plt.close()

    # plot the histogram of gray values in the image
        fig, ax = plt.subplots(1, 2, figsize = (8, 3))
        ax[0].imshow(grayscale_image, cmap = plt.cm.gray)
        ax[0].set_axis_off()
        ax[0].set_title('Inverted Initial Grayscale Image', fontsize = 8)

        ax[1].plot(hist_centers, hist, lw = 2)
        ax[1].set_title('Histogram of Gray Values', fontsize = 8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{outdir}/Intermediate_Plots/{image_name}.Grayscale_Histogram.png', dpi = 300)
        plt.close()

    return grayscale_image, initial_image, image_name, min_hist_val


def gaussian_filter(grayscale_image, sigma: float = 6.0, truncate = 3.0):
    img_gau_filt = ski.filters.gaussian(grayscale_image, sigma = sigma, truncate = truncate)
    return img_gau_filt

def segment_cells(
        grayscale_image: arr,
        min_hist_val: int,
        outdir: str,
        image_name: str,
        make_plots: bool = True,
        verbose: bool = True
    ) -> arr:
    """
    Segments the grayscale image.

    Parameters
    ----------
    grayscale_image: nd array of the processed image
    outdir:          output directory to store output data
    image_name:      name of the image without its extension
    make_plots:      include plots of intermediate data/steps
    verbose:         be loud and proud

    Returns
    ----------
    segmented_cells: nd array of the processed image
    """
    #--- region-based segmentation ---#
    # make an elevation map using the Sobel gradient
    elevation_map = ski.filters.sobel(grayscale_image)

    if make_plots:
        # plot the evelation map
        fig, ax = plt.subplots(figsize = (4, 3))
        ax.imshow(elevation_map, cmap = plt.cm.gray)
        ax.set_title('elevation map')
        ax.set_axis_off()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{outdir}/Intermediate_Plots/{image_name}.Elevation_Map.png')
        plt.close()

    # Find markers of the background and cells based on the extremes in the histogram of
    # gray scale values
    markers = np.zeros_like(grayscale_image)
    markers[grayscale_image < 90] = 1
    markers[grayscale_image > 90] = 2

    # if make_plots:
    #     fig, ax = plt.subplots(figsize=(4, 3))
    #     ax.imshow(markers, cmap=plt.cm.nipy_spectral)
    #     ax.set_title('markers')
    #     ax.set_axis_off()
    #     plt.show()

    # Use watershed segmentation to fill regions of the elevation map
    segmented_cells = ski.segmentation.watershed(elevation_map, markers)

    plt.imshow(segmented_cells)
    plt.show()

    if make_plots:
        fig, ax = plt.subplots(figsize = (4, 3))
        ax.imshow(segmented_cells, cmap = plt.cm.gray)
        ax.set_title(f'Initial Segmentation of {image_name}', fontsize = 8)
        ax.set_axis_off()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{outdir}/Intermediate_Plots/{image_name}.Initial_Segmentation.png')
        plt.close()

    return segmented_cells


def clean_segmentation(
        segmented_cells: arr,
        outdir: str,
        image_name: str,
        make_plots: bool = True,
        verbose: bool = True
    ) -> arr:
    """
    Fills in and "cleans" the initial segmented cells.

    Parameters
    ----------
    segmented_cells: nd array of the initial segmented cells
    outdir:          output directory to store output data
    image_name:      name of the image without its extension
    make_plots:      include plots of intermediate data/steps
    verbose:         be loud and proud

    Returns
    ----------
    cells_cleaned: nd array of filled in and "cleaned" segmented image
    """
    # Fill in the segmented cells using mathematical morphology
    filled_segmented_cells = ndi.binary_fill_holes(segmented_cells - 1)

    # Ignore any cells that "run-off" the edge of the image
    clear_segmented_cells = ski.segmentation.clear_border(filled_segmented_cells)

    # Remove spurious fills
    cells_cleaned = ski.morphology.remove_small_objects(clear_segmented_cells, min_size = 300)

    if make_plots:
        fig, ax = plt.subplots(figsize = (4, 3))
        ax.imshow(cells_cleaned, cmap = plt.cm.gray)
        ax.set_title(f'Cleaned and Segmented Cells from: {image_name}')
        ax.set_axis_off()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{outdir}/Intermediate_Plots/{image_name}.Clean_Segmentation.png')
        plt.close()

    return cells_cleaned


def label_segmented_cells(
        initial_image: arr,
        cells_cleaned: arr,
        outdir: str,
        image_name: str,
        verbose: bool = True
        ) -> None:
    """
    Labels the segmented cells, generates the final plot, creates a table of counts?

    Parameters
    ----------
    initial_image:  nd array of the inital provided image
    cells_cleaned:  nd array of filled in and "cleaned" segmented image
    outdir:         output directory to store output data
    image_name:     name of the image without its extension
    make_plots:     include plots of intermediate data/steps
    verbose:        be loud and proud

    Returns
    ----------
    UNSURE:
    """

    # Label the cells
    labeled_cells, _ = ndi.label(cells_cleaned)

    # get the final cell count
    final_cells, final_cell_num = ski.measure.label(labeled_cells > 0, return_num = True)

    image_label_overlay = ski.color.label2rgb(labeled_cells, image = initial_image[:,:,:3], bg_label = 0, image_alpha = 0.6)

    fig, ax = plt.subplots(1, 2, figsize = (8, 3), sharey = True)
    ax[0].imshow(initial_image[:,:,:3], cmap = plt.cm.gray)
    ax[0].contour(cells_cleaned, [0.5], linewidths = 0.4, colors='magenta')
    ax[1].imshow(image_label_overlay)
    ax[1].contour(cells_cleaned, [0.5], linewidths = 0.1, colors='black')

    for a in ax:
        a.set_axis_off()

    # fig.set_title(f'Cells Identified and Counted from {image_name}')
    # plt.tight_layout()
    # plt.savefig(f'{outdir}/Final_Plots/{image_name}.Labeled_Cells.png', dpi = 300)

    for region in ski.measure.regionprops(final_cells):
        # Take the centroid of the region and use it for placing the label
        y, x = region.centroid
        plt.text(x, y+30, f"Cell: {region.label}", color = 'black', fontsize = 4, ha = 'center', va = 'center')


    plt.suptitle(f'Cells Identified and Counted from {image_name}', fontsize = 8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f'{outdir}/Final_Plots/{image_name}.Labeled_Counted_Cells.png', dpi = 300)
    plt.close()

    return final_cells


def extract_bbox_per_cell(
        final_cells,
        initial_image,
        image_name,
        outdir):
    tmp_img_outdir = f'{outdir}/Cropped_Cell_Images/'
    Path(tmp_img_outdir).mkdir(exist_ok = True, parents = True)

    out_crop_prefix = f'{tmp_img_outdir}{image_name}.Cell_'

    for region in ski.measure.regionprops(final_cells):
        # Get cell label
        cell_label = region.label

        # Retrieve bounding box coordinates
        min_xpos, min_ypos, max_xpos, max_ypos = region.bbox

        # Crop the image itself to the bounding box
        cell_crop_bbox = initial_image[:,:,:3][min_xpos:max_xpos, min_ypos:max_ypos]

        ski.io.imsave(f'{out_crop_prefix}{cell_label}.png', cell_crop_bbox)


def parse_metadata(image_file: str):
    """
    Parses the metadata to infer the um^2 per pixel

    Parameters
    ----------
    image_file:  path to the image to process

    Returns
    ----------
    area per pixel in micrometers
    """
    if '.ome.' not in image_file:
        return None

    physx_pixel, physy_pixel = None, None

    with tifffile.TiffFile(image_file) as tif:
        ome_metadata = tif.ome_metadata
        # print(ome_metadata.split('PhysicalSizeX="')[1])
        physx_pixel = float(ome_metadata.split('PhysicalSizeX="')[1].partition('"')[0])

        # Temporary FIX!
        # if 'Model="PLAN APO λD 20x OFN25 DIC N2"' in ome_metadata:
        #     physx_pixel *= 5

    # return microns per pixel
    return physx_pixel


def calc_cell_stats(
    final_cells: arr,
    img_file: str,
    outdir: str,
    image_name: str,
    verbose: bool = True
    ):

    # store (potentially) helpful cell measurements
    labeled_cell_info = {}

    um_per_pixel = parse_metadata(img_file)

    for region in ski.measure.regionprops(final_cells):
        labeled_cell_info[region.label] = {}
        labeled_cell_info[region.label]['Cell_Area'] = f'{region.area * um_per_pixel**2:.2f}'
        labeled_cell_info[region.label]['Cell_Perimeter'] = f'{region.perimeter * um_per_pixel:.2f}'
        labeled_cell_info[region.label]['Cell_Eccentricity'] = f'{region.eccentricity:.4f}'
        labeled_cell_info[region.label]['Aspect_Ratio'] = f'{(region.axis_major_length * region.axis_minor_length * um_per_pixel**2):.2f}'
        labeled_cell_info[region.label]['Cell_Length'] = f'{region.axis_major_length * um_per_pixel:.2f}'
        labeled_cell_info[region.label]['Cell_Width'] = f'{region.axis_minor_length * um_per_pixel:.2f}'
        labeled_cell_info[region.label]['Cell Solidity'] = f'{region.solidity:.2f}'
        labeled_cell_info[region.label]['Cell Convex Area'] = f'{region.convex_area:.2f}'
        labeled_cell_info[region.label]['Image_File'] = f'{image_name}'

    df = pd.DataFrame(labeled_cell_info).T
    df.index.name = 'Cell_Number'
    df.to_csv(f'{outdir}/{image_name}.Cell_Stats.csv')


def process_image_files(
        input_img: str,
        outdir: str,
        make_plots: bool = True,
        verbose: bool = True
        ) -> None:

    image_extensions = ['png','tiff','tif','jpg']
    image_files = []

    cell_counts = {}
    all_cell_info = {}

    if input_img.rpartition(".")[-1].lower() in image_extensions:
        image_files.append(input_img)

    else:
        for ext in image_extensions:
            image_files += glob.glob(f'{input_img}/*{ext}')

    for img in image_files:
        if verbose:
            print(f'Assessing image: {img.rpartition("/")[-1]}')

        gs_image, init_img, img_name, min_hist_val = prepare_img_file(
                                                        img,
                                                        outdir,
                                                        make_plots,
                                                        True,
                                                        verbose
                                                        )

        seg_cells = segment_cells(
                        gs_image,
                        min_hist_val,
                        outdir,
                        img_name,
                        make_plots,
                        verbose
                        )

        clean_cells = clean_segmentation(
                        seg_cells,
                        outdir,
                        img_name,
                        make_plots,
                        verbose
                        )

        final_cells = label_segmented_cells(
                        init_img,
                        clean_cells,
                        outdir,
                        img_name,
                        verbose
                        )

        extract_bbox_per_cell(
                final_cells,
                init_img,
                img_name,
                outdir)

        calc_cell_stats(
            final_cells,
            img,
            outdir,
            img_name,
            verbose
            )


if __name__ == '__main__':
    try:
        input_img = sys.argv[1]
        outdir = sys.argv[2]
    except:
        print('\nUsage:\n\n    python3 tetra_count.py [IMAGE-FILE-OR-DIRECTORY] [OUTPUT-DIRECTORY]\n\n')
        sys.exit(1)

    verbose = True
    make_plots = True

    # print('\n\nAssuming the metadata objective is truly 4x not 20x.\nAdjusting um per pixel accordingly.\n\nIF THIS IS WRONG, COMMENT LINES 309/310 out!!!\n')

    process_image_files(
        input_img,
        outdir,
        make_plots,
        verbose
        )
