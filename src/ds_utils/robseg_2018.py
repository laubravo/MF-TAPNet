import os
import cv2
import numpy as np
import tqdm
from pathlib import Path
import argparse


"""
utils for MICCAI 2018 robot instrument segmentation dataset
"""

num_frames_video = 149 # number of frames per video
original_height, original_width = 1024, 1280 # original img h, w


num_videos = {
    'train': 8,
    'test': 10,
}

# self defined train_val split
folds = {0: [1, 3],
         1: [2, 5],
         2: [4, 8],
         3: [6, 7]}

# number of classes for each type of problem
problem_class = {
    'binary': 2,
    'parts': 4,
    'instruments': 8
}

# factor
problem_factor = {
    'binary' : 255,
    'parts' : 85,
    'instruments' : 32
}

# folder names
mask_folder = {
    'binary': 'binary_masks',
    'parts': 'parts_masks',
    'instruments': 'instruments_masks'
}

# for linear interpolation
mask_folder_linear = {
    'binary': 'binary_masks_linear',
    'parts': 'parts_masks_linear',
    'instruments': 'instruments_masks_linear'
}


"""
file utilities:
get images, masks, optical flows, predictions, attention maps 
from specific instrument dataset
"""


def get_data(data_dir, data_type, mode='train', folder_id=-1):
    """
    @params:
    data_type: image, optflows or problem type

    return:
    a list of data of specific type
    """

    global num_videos, mask_folder

    num_folders = num_videos[mode]
    
    # if it belongs to problem_type, return masks
    if data_type in mask_folder.keys():
        folder_name = mask_folder[data_type]
    else:
        folder_name = data_type

    filenames = []

    if folder_id > 0:
        # get instrument_dataset_<folder_id>
        filenames += (Path(data_dir) / ('instrument_dataset_' + str(folder_id)) \
            / folder_name).glob('*')
    elif folder_id == -1:
        # get all data
        for folder_id in range(1, num_folders + 1):
            filenames += (Path(data_dir) / ('instrument_dataset_' + str(folder_id)) \
            / folder_name).glob('*')

    if len(filenames) <= 0:
        raise ValueError("Empty folder, data_type: %s, mode: %s, folder_id: %d"
            % (data_type, mode, folder_id))
    
    # sort by name
    return list(sorted(filenames))


# TODO
def get_preds(preds_dir, problem_type, folder_id):
    '''
    preds_dir: Pathlib Path object
    '''
    filenames = list(sorted((preds_dir / ('instrument_dataset_' + str(folder_id)) / problem_type).glob('*')))
    return filenames

def get_all_preds(preds_dir, problem_type):
    '''
    preds_dir: Pathlib Path object
    '''
    filenames = []
    for folder_id in range(1, 9):
        filenames += get_preds(preds_dir, problem_type, folder_id)
    return filenames


"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""

def preprocess_data(args):
    """
    preprocess data following the instructions
    * for each problem type, squeeze the gt to one two-dim grayscale mask scaled to [0, 255]

    @param: argparse args
    """

    # data dir
    data_dir = Path(args.data_dir)
    assert data_dir.exists() == True

    # cropped data dir
    target_data_dir = Path(args.target_data_dir)
    target_data_dir.mkdir(exist_ok=True, parents=True)

    global num_videos, problem_factor, mask_folder

    num_folders = num_videos[args.mode] # number of folders

    for idx in range(1, num_folders + 1):
        # original dataset dir
        instrument_folder = data_dir / ('instrument_dataset_' + str(idx))

        # video frames dir (only read left frames)
        frames_dir = instrument_folder / 'left_frames'

        # processed dataset dir
        processed_instrument_folder = target_data_dir / ('instrument_dataset_' + str(idx))

        # mkdir for each problem_type
        image_folder = processed_instrument_folder / 'images'
        image_folder.mkdir(exist_ok=True, parents=True)

        if args.mode == 'train':
            # original mask folder
            ori_mask_folders = list((instrument_folder / 'ground_truth').glob('*'))

            # new mask folder
            instrument_mask_folder = processed_instrument_folder / mask_folder['instruments']
            instrument_mask_folder.mkdir(exist_ok=True, parents=True)

        for file_name in tqdm.tqdm(list(frames_dir.glob('*')),
            desc='preprocess dataset %d' % idx, dynamic_ncols=True):
            img = cv2.imread(str(file_name))
            h, w, _ = img.shape

            # save cropped frame
            cv2.imwrite(str(image_folder / (file_name.name)), img)

            if args.mode == 'test':
                continue # test data has no masks

            # create empty masks
            mask_instruments = np.zeros((h, w))

            for ori_mask_folder in ori_mask_folders:
                # read in grayscale
                mask = cv2.imread(str(ori_mask_folder / file_name.name), 0)

                # mark each type of instruments
                # background will be set to 0 in default
                if 'Bipolar_Forceps' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 1
                elif 'Prograsp_Forceps' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 2
                elif 'Large_Needle_Driver' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 3
                elif 'Vessel_Sealer' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 4
                elif 'Grasping_Retractor' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 5
                elif 'Monopolar_Curved_Scissors' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 6
                elif 'Other' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 7

            # crop and save masks
            cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)
