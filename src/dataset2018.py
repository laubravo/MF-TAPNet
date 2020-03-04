import os
import os.path as osp
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from albumentations.pytorch.functional import img_to_tensor
from albumentations import Resize
import random
# modules
import ds_utils.robseg_2018 as utils
from attmap_utils import init_attmaps_np, cal_attmap_np
import pdb
from glob import glob


class RobotSegDataset2018(Dataset):
    """docstring for RobotSegDataset2018"""
    def __init__(self, data_dir, transform, mode, model,
        problem_type, semi=False, **kwargs):
        super(RobotSegDataset2018, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type
        self.factor = utils.problem_factor[problem_type]
        self.mask_folder = 'annotations'
        self.model = model
        self.semi = semi
        self.filenames = glob(osp.join(self.data_dir, 'images','*'))

        # for TAPNet, change dataset schedule for training
        if 'TAPNet' in self.model:
            self.mf = kwargs['mf']
            if self.mode == 'train':
                self.batch_size = kwargs['batch_size']
                # init dataset schedule to be ordered
                self.set_dataset_schedule('ordered')
            # init attention maps
            self.init_attmaps()


    def __len__(self):
        # num of imgs
        return len(self.filenames)

    def __getitem__(self, idx):
        '''
        pytorch dataloader get_item_from_index
        input:
        idx: corresponding with __len__

        output:
        input_dict: a dictionary stores all return value
        '''

        # input dict for return
        input_dict = {}
        filename, abs_idx = self.get_filename_from_idx(idx)
        image = self.load_image(filename)

        # extra input for TAPNet
        if 'TAPNet' in self.model:
            # load optical flow <prev_frame, cur_frame>
            optflow = self.load_optflow(filename)
            # generate attention map
            if abs_idx % utils.num_frames_video:
                if self.mf:
                    # calculate attention map using previous prediction and Motion Flow
                    attmap = cal_attmap_np(self.attmaps[abs_idx - 1], optflow)
                else:
                    # calculate attention map using previous prediction
                    attmap = self.attmaps[abs_idx - 1]

                '''
                EXPERIMENT: uncomment to use
                don't use optical flow, directly use previous frame prediction in last epoch
                '''
                # attmap = self.attmaps[abs_idx - 1]
            else:
                # first frame of every video, simply use prediction in last epoch without motion flow
                attmap = self.attmaps[abs_idx]

            # input absolute index and attention map for attention map update
            input_dict['abs_idx'] = abs_idx
            input_dict['attmap'] = torch.from_numpy(np.expand_dims(attmap, 0)).float()

        # gts
        mask = self.load_mask(filename, self.mask_folder)

        # augment
        data = {'image': image, 'mask': mask}

        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        # input image
        input_dict['input'] = img_to_tensor(image)
        if self.mode == 'eval':
            # in evaluation mode should input the filename of input image
            input_dict['input_filename'] = str(filename)
        else:
            # input gt
            input_dict['target'] = torch.from_numpy(mask).long()

        return input_dict

    def mark_labels(self, method):
        '''
        mark labeles for dataset
        the first and last label should be 1 
        e.g. semi_percentage = 0.3

        interval: labeled = 1,0,0,1,0,0,1......1
        random: randomly mark semi_percentage of data as labeled
        '''
        num_files = len(self.filenames)
        ratio = self.semi_percentage
        assert num_files > 2
        if method == 'interval':
            labeled = np.zeros((num_files,))
            sep = int(1 // ratio)
            labeled[::sep] = 1
            labeled[-1] = 1
        elif method == 'random':
            labeled = np.random.choice([0,1], size=num_files, p=[1-ratio, ratio])
        else:
            raise NotImplementedError

        return labeled



    def get_filename_from_idx(self, idx):
        '''
        get filename from pytorch shuffled index

        input:
        pytorch dataset __getitem__ idx

        output:
        filename: correct filename in specific order
        abs_idx: absolute index in original dataset order
        '''

        filename = self.filenames[idx]
        abs_idx = idx
        if 'TAPNet' in self.model and self.mode == 'train':
            if self.schedule == "shuffle":
                filename = self.shuffled_filenames[idx]
                abs_idx = self.shuffled_idx[idx]
            elif self.schedule == "ordered":
                filename = self.ordered_filenames[idx]
                abs_idx = self.ordered_idx[idx]
            else:
                raise NotImplementedError
        return filename, abs_idx

    def update_attmaps(self, outputs, abs_idxs):
        '''
        Update attention maps using current predictions

        outputs: <b, c, h, w> ndarray softmax(output)
        abs_idxs: absolute ndarray indices in original dataset order
        '''

        assert outputs.ndim == 4
        assert abs_idxs.ndim == 1
        assert outputs.shape[0] == abs_idxs.shape[0]

        # update attention map using prediction for current frame
        b, c, h, w = outputs.shape
        # probability of being non-background
        self.attmaps[abs_idxs] = 1 - outputs[:,0,:,:]

    def load_image(self, filename):
        return cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)

    def load_mask(self, filename, mask_folder):
        # change dir name
        mask = cv2.imread(str(filename).replace('images', mask_folder), 0)

        return (mask).astype(np.uint8)

    def load_optflow(self, filename):
        # read .flo file
        with open(str(filename).replace('images', 'optflows').replace('png', 'flo')) as f:
            header = np.fromfile(f, dtype=np.uint8, count=4)
            size = np.fromfile(f, dtype=np.int32, count=2)
            optflow = np.fromfile(f, dtype=np.float32) \
                .reshape(utils.original_height, utils.original_width, 2)# .transpose(2,0,1)

            #h, w = utils.cropped_height, utils.cropped_width
            # if applied resize operation, resize it
            for transform in self.transform.transforms:
                if isinstance(transform, Resize):
                    h, w = transform.height, transform.width
                    break
            # TODO: change this for better resize
            optflow = cv2.resize(optflow, dsize=(w, h))

        return optflow

    # dataset schedule [ordered, shuffle]
    def set_dataset_schedule(self, schedule):
        self.schedule = schedule
        if self.schedule == "shuffle":
            self.shuffle_dataset()
        elif self.schedule == "ordered":
            if not hasattr(self, "ordered_filenames"):
                self.init_order()
        else:
            raise NotImplementedError

    # random shuffle the index of filenames
    def shuffle_dataset(self):
        self.shuffled_idx = np.arange(0, len(self.filenames))
        np.random.shuffle(self.shuffled_idx)
        self.shuffled_filenames = [self.filenames[idx] for idx in self.shuffled_idx]

    # define order of filenames to be trained
    def init_order(self):
        self.ordered_idx = []
        # stats
        num_of_files = len(self.filenames)
        num_frames_video = utils.num_frames_video
        num_videos = num_of_files // num_frames_video
        num_batches = num_frames_video // self.batch_size

        for i in range(num_batches):
            for j in range(self.batch_size):
                for dataset_idx in range(num_videos):
                    self.ordered_idx.append(dataset_idx * num_frames_video + j * num_batches + i)
        for j in range(self.batch_size * num_batches, num_frames_video):
            for dataset_idx in range(num_videos):
                self.ordered_idx.append(dataset_idx * num_frames_video + j)
        self.ordered_idx = np.array(self.ordered_idx, dtype=int)
        self.ordered_filenames = [self.filenames[idx] for idx in self.ordered_idx]

    def init_attmaps(self):
        # init attention maps for each frame
        num_imgs = len(self.filenames)
        #h, w = utils.cropped_height, utils.cropped_width
        # if applied resize operation, resize it
        for transform in self.transform.transforms:
            if isinstance(transform, Resize):
                h, w = transform.height, transform.width
                break

        # init with zero attention
        self.attmaps = init_attmaps_np(num_imgs, h, w)
        # self.attmaps = np.random.randn(num_imgs, h, w)
