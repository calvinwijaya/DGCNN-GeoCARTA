# Source: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/indoor3d_util.py

import numpy as np
import numpy as np
import glob
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------

min_point_discard_block = 100

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)


def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label

def room2blocks(data, label, num_point, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1, use_all_points=False):
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        # print('num_block_x', num_block_x)
        # print('num_block_y', num_block_y)
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * stride)
                ybeg_list.append(j * stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)
    
    # Collect blocks
    block_data_list = []
    block_label_list = []
    global raw_data_index
    idx = 0
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]

        # to check which data belong to which block
        xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
        ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < min_point_discard_block:  # discard block if there are less than 20 pts (original 100 pts).
            continue

        # select data according to the boundary of the block
        block_data = data[cond, :]
        block_label = label[cond]

        if use_all_points:
            block_data_list.append(block_data)
            block_label_list.append(block_label)
        else:
            # randomly subsample data
            block_data_sampled, block_label_sampled = \
                sample_data_label(block_data, block_label, num_point)
            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))

    if use_all_points:
        block_data_return, block_label_return = np.array(block_data_list), np.array(block_label_list)
    else:
        block_data_return, block_label_return = np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)

    # return np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)
    return block_data_return, block_label_return

def room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                random_sample, sample_num, sample_aug):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:, 0:-1]
    label = data_label[:, -1].astype(np.uint8) # The label, always in the last column!!
    
    data[:, 3:6] /= 255.0 # normalized rgb color

    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])

    data_batch, label_batch = room2blocks(data, label, num_point, block_size, stride,
                                          random_sample, sample_num, sample_aug)

    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9)) # XYZ, RGB, normXYZ
    for b in range(data_batch.shape[0]):
        
        # add normalized XYZ (column 6, 7, 8)
        new_data_batch[b, :, 6] = data_batch[b, :, 0] / max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2] / max_room_z
        
        # recenter for each block
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx + block_size / 2)
        data_batch[b, :, 1] -= (miny + block_size / 2)

    # Colummn: XYZ, RGB, normXYZ
    new_data_batch[:, :, 0:6] = data_batch[:,:,0:6] # for XYZ, RGB
    
    return new_data_batch, label_batch


def room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0,
                                   random_sample=False, sample_num=None, sample_aug=1):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                       random_sample, sample_num, sample_aug)