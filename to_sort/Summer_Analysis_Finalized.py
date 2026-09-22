#!/usr/bin/env python3

"""
Script to perform the base segementation for yeast images. This includes automated
cell detection, segementation, labeling and measurement extraction.

Largely based on initial code by Maisie Wang (W&M '27).

Dependencies:
- python >= 3.10
- matplotlib
- pandas
- sci-kit image >= 0.26.0
- scipy
"""

# import everything:
import glob, sys
from collections import defaultdict # might not be necessary

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

import pandas as pd

from scipy import ndimage as ndi

import skimage as ski


def prep_image(image_path: str) -> list:
    """
    Something goes here
    """
    img = ski.io.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print("Image loaded successfully!")

    gray_image = ski.color.rgb2gray(img[:,:,:3])
    grayscale_image =  ski.util.invert(ski.util.img_as_ubyte(gray_image))

    return grayscale_image


def img_gaussian_filter(grayscale_image, sigma: float = 1.5):#, truncate: float = 3.0):
    """
    Something goes here
    """
    img_gau_filt = ski.filters.gaussian(grayscale_image, sigma = sigma)#, truncate = truncate)
    return img_gau_filt


def segment_image(
        img_gau_filt,
        img_name: str,
        radius: int = 8,
        max_hole_size: int = 200,
        min_obj_size: int = 400,
        make_plots: bool = False):
    # reformat the picture
    img_for_mask = ski.util.img_as_ubyte(img_gau_filt)

    # set up the image "repairing" to remove holes/close cells
    footprint = ski.morphology.disk(radius)

    # apply a local OTSU threshold
    local_otsu = ski.filters.rank.otsu(img_for_mask, footprint)

    # Run the initial masking
    # init_img_mask = img_for_mask >= local_otsu

    # Li thresholding below... Tends to undercount compared to local OTSU (~60% of local-otsu)
    init_img_mask = img_for_mask >= ski.filters.threshold_li(img_for_mask)

    # fill the small holes in the initial masking
    mask_holes_filled = ~ski.morphology.remove_small_holes(init_img_mask, max_size = max_hole_size)

    # fill the remaining objects
    mask_binary_fill = ndi.binary_fill_holes(mask_holes_filled)

    # remove small filled objects (artifacts from the preparation)
    final_cell_mask = ski.morphology.remove_small_objects(mask_binary_fill, max_size = min_obj_size)

    # remove any cells that are touching the image's "border"
    cleared_final_mask = ski.segmentation.clear_border(final_cell_mask)

    # makes plots if desired
    if make_plots:
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(img_for_mask, cmap = plt.cm.gray)
        ax[0, 0].set_title('Grayscale + Gaussian-Blurred')

        ax[0, 1].imshow(init_img_mask, cmap = plt.cm.gray)
        ax[0, 1].set_title('Initial masking -- No "filling"')

        ax[1, 0].imshow(final_cell_mask, cmap = plt.cm.gray)
        ax[1, 0].set_title('Final masking -- With "filling"')

        ax[1, 1].imshow(cleared_final_mask, cmap = plt.cm.gray)
        ax[1, 1].set_title('Final masking -- cleared border')
        plt.tight_layout()

        plt.savefig(f'{img_name.rpartition(".")[0]}.Segmentation.png', dpi = 300)

    return cleared_final_mask


def calc_circularity(area: float, perimeter: float) -> float:
    return (4 * np.pi * area) / (perimeter ** 2)


def eval_user_inputs():
    keep_cell_list = ['yes', 'y', 'no', 'n', 'stop']
    cell_type_list = ['single', 'large', 'large-budded', 'other', 'stop']

    user_keep_cell = ''
    user_cell_type = ''

    while not user_keep_cell.lower() in keep_cell_list:
        user_keep_cell = input('Keep this cell? [yes / no]:  ')

    if user_keep_cell.lower() == 'stop':
        return ['STOP', 'STOP']

    if user_keep_cell.lower() in ['no', 'n']:
        return ['Skipped', 'Skipped']

    else:
        while not user_cell_type.lower() in cell_type_list:
            user_cell_type = input('What type of cell is this [single, large, large-budded, other]:  ')
            if user_cell_type.lower() == 'stop':
                return ['STOP', 'STOP']

        return ['Analyzed', user_cell_type.lower()]


def run_interactive_plotting(
        labeled_image,
        grayscale_image,
        img_file_name,
        min_circularity: float = 0.7,
        max_area: int = 20000):
    plt.ion()
    fig, ax = plt.subplots()
    plt.show()

    labeled_cell_dict = defaultdict(dict)

    good_regions = []

    for region in ski.measure.regionprops(labeled_image, intensity_image = grayscale_image):
        keep_cell = False
        check_cell = True
        exit_all = False

        circularity = calc_circularity(region.area, region.perimeter)

        if circularity >= min_circularity and region.area <= max_area:
            keep_cell = True
            good_regions.append(region.coords)

            # bounding box coords
            minr, minc, maxr, maxc = region.bbox

            if keep_cell == True:

                ax.imshow(grayscale_image, cmap = plt.cm.gray)

                rect = mpatches.Rectangle(
                    (minc-10, minr-10),
                    (maxc - minc)+20,
                    (maxr - minr)+20,
                    fill = False,
                    edgecolor = 'red',
                    linewidth = 3)

                ax.add_patch(rect)
                ax.set_axis_off()

                plt.tight_layout()

                plt.gcf().canvas.draw()
                
                bbox_cell_img = f'{img_file_name.rpartition("/")[-1].rpartition(".")[0]}.Cell_{region.label:03}.png'
                print(bbox_cell_img)

                plt.savefig(bbox_cell_img, dpi  300)

                while check_cell == True:
                    plt.pause(0.1)

                    user_summary = eval_user_inputs()

                    if 'STOP' in user_summary:
                        exit_all = True
                        break

                    # bbox_cell_img = f'{img_file_name.rpartition("/")[-1].rpartition(".")[0]}.Cell_{region.label:03}.png'
                    # print(bbox_cell_img)
                    #
                    # plt.savefig(bbox_cell_img, dpi  300)

                    labeled_cell_dict[f'Cell-{region.label:03}'] = {
                        'Cell-Evaluated':user_summary[0],
                        'Cell-Type':user_summary[1],
                        'Perimeter': f'{region.perimeter:.3f}',
                        'Area': f'{region.area:.3f}',
                        'Perimeter-Crofton': f'{region.perimeter_crofton:.3f}',
                        'Eccentricity': f'{region.eccentricity:.3f}',
                        'Convex-Area': f'{region.area_convex:.3f}',
                        'Equivalent-Diameter': f'{region.equivalent_diameter_area:.3f}',
                        'Mean-Intensity': f'{region.intensity_mean:.3f}',
                        'Circularity': f'{circularity:.3f}',
                        'Image_File': img_file_name}

                    ax.cla()

                    check_cell = False

                if exit_all == True:
                    break

    plt.close('all')

    make_fake_color(labeled_image, good_regions, grayscale_image, img_file_name)

    return labeled_cell_dict


def label_segmented_cells(
        img_file_name,
        segmented_image,
        grayscale_image,
        min_circularity: float = 0.7,
        max_area:int = 20000,
        interactive_plot: bool = True):
    # label the image
    labeled_image = ski.measure.label(segmented_image)

    # Properties to keep ... cell_props = ['label', 'area', 'bbox', ]

    if interactive_plot:
        labeled_cell_dict = run_interactive_plotting(
            labeled_image,
            grayscale_image,
            img_file_name,
            min_circularity,
            max_area)

    else:
        good_regions = []
        labeled_cell_dict = defaultdict(dict)

        for region in ski.measure.regionprops(labeled_image, intensity_image = grayscale_image):
            circularity = calc_circularity(region.area, region.perimeter)

            if circularity >= min_circularity and region.area <= max_area:
                good_regions.append(region.coords)

                labeled_cell_dict[f'Cell-{region.label:03}'] = {
                    'Perimeter': f'{region.perimeter:.3f}',
                    'Area': f'{region.area:.3f}',
                    'Perimeter-Crofton': f'{region.perimeter_crofton:.3f}',
                    'Eccentricity': f'{region.eccentricity:.3f}',
                    'Convex-Area': f'{region.area_convex:.3f}',
                    'Equivalent-Diameter': f'{region.equivalent_diameter_area:.3f}',
                    'Mean-Intensity': f'{region.intensity_mean:.3f}',
                    'Circularity': f'{circularity:.3f}',
                    'Image_File': img_file_name}

        print(len(good_regions))

        make_fake_color(labeled_image, good_regions, grayscale_image, img_file_name)

    return labeled_cell_dict


def make_fake_color(
        labeled_image,
        good_regions,
        grayscale_image,
        img_name):

    out_img_name = f'{img_name.rpartition(".")[0]}.Cells_Analyzed.png'

    color_labeled_cell_array = np.zeros_like(labeled_image, dtype = np.int32)

    for reg in good_regions:
        for row, col in reg:
            color_labeled_cell_array[row][col] = labeled_image[row][col]

    image_label_overlay = ski.color.label2rgb(color_labeled_cell_array, image = grayscale_image, bg_label = 0)

    plt.imshow(image_label_overlay)

    plt.savefig(out_img_name, dpi = 300)

    plt.clf()


def save_to_table(image_analysis_summary: dict, output_name: str):
    df = pd.DataFrame(image_analysis_summary).T
    df.index.name = 'Cell-Number'
    df.to_csv(output_name)


def segment_measure_cells(
        all_images: list,
        sigma: float = 6.0,
        radius: int = 35,
        max_area: int = 20000,
        max_hole_size: int = 100,
        min_obj_size: int = 200,
        min_circularity: float = 0.7,
        make_plots: bool = True,
        interactive: bool = False):

    image_analysis_summary = {}

    # Analyze all the images!
    for img_file in all_images:
        img_file_name = img_file.rpartition("/")[-1]
        print(f'Preparing image: {img_file_name}!')

        grayscale_image = prep_image(img_file)

        img_gau_filt = img_gaussian_filter(
            grayscale_image,
            sigma)

        print(f'Segmenting image: {img_file_name}!')
        segmented_image = segment_image(
            img_gau_filt,
            img_file_name,
            radius,
            max_hole_size,
            min_obj_size,
            make_plots)

        img_analysis_dict = label_segmented_cells(
            img_file_name,
            segmented_image,
            grayscale_image,
            min_circularity,
            max_area,
            interactive)

        image_analysis_summary.update(img_analysis_dict)

    return image_analysis_summary


def grab_image_files(image_path: str):
    # Allow analyzing PNG or TIFF images
    valid_extensions = ['.png','.tiff']
    all_images = []

    for ve in valid_extensions:
        all_images += glob.glob(f'{image_path.rstrip(ve)}*{ve}')

    return all_images


if __name__ == '__main__':
    try:
        image_path = sys.argv[1]
        interactive_plotting = sys.argv[2]

    except IndexError:
        print('\n\nUsage:\n\n    python3 Summer_Analysis_Finalized.py [PATH-TO-IMAGE(S)] [INTERACTIVE -- YES OR NO]\n\n')
        sys.exit()

    if interactive_plotting.lower() not in ['yes', 'y', 'no', 'n']:
        print('\n\nUsage:\n\n    python3 Summer_Analysis_Finalized.py [PATH-TO-IMAGE(S)] [INTERACTIVE -- YES OR NO]\n\n')
        sys.exit()

    elif interactive_plotting in ['yes','y']:
        interactive = True

    else:
        interactive = False

    # reasonable parameters from Xyrus's perspective (based on one image)
    sigma = 5
    radius = 35
    max_hole_size = 100
    min_obj_size = 200
    max_area = 20000
    min_circularity = 0.7
    make_plots = True

    output_csv = f'{image_path.rpartition("/")[-1].rpartition(".")[0]}.csv'

    all_images = grab_image_files(image_path)

    image_analysis_summary = segment_measure_cells(
        all_images,
        sigma,
        radius,
        max_area,
        max_hole_size,
        min_obj_size,
        min_circularity,
        make_plots,
        interactive)

    save_to_table(image_analysis_summary, output_csv)
