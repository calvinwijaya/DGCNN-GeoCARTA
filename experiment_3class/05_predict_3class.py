# 6. Define testing function

CUDA_LAUNCH_BLOCKING="1"

"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import shutil
from data_utils.dataLoader import ScannetDatasetWholeScene
from models.dgcnn_sem_seg import dgcnn_sem_seg
import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys
import numpy as np
import laspy
import time
import re

from data_utils.split_merge_las import *

BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['0', '1', '2']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def read_las(las_files):
    inFile = laspy.read(las_files)
    x = inFile.x
    y = inFile.y
    z = inFile.z
    r = inFile.red
    g = inFile.green
    b = inFile.blue
    
    # Add Intensity
    I = inFile.intensity
    
    data = np.column_stack((x, y, z, r, g, b, I))
    
    return data

def save_las(X, filename):
    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)
    las.x = X[:, 0]
    las.y = X[:, 1]
    las.z = X[:, 2]
    las.red = X[:, 3]
    las.green = X[:, 4]
    las.blue = X[:, 5]
    las.intensity = X[:, 6]
    las.classification = X[:, 7]
    las.write(filename + ".las")

def parse_args(test_area=None):
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=test_area, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--num_classes', type=int, default=3, help='How many classes used for segmentation [default: 9]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--point_cloud', type=str, required=True, help='Name of point cloud data')
    parser.add_argument('--block_size', type=int, default= 1000, help='Size of each block')
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                if int(pred_label[b, n]) < vote_label_pool.shape[1]:
                    vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
                else:
                    print(f"Warning: Predicted label {int(pred_label[b, n])} is out of bounds for the current class range.")
    return vote_label_pool

def main(args):
    # def log_string(str):
    #     logger.info(str)
    #     print(str)
    
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    # logger = logging.getLogger("Model")
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    # log_string('PARAMETER ...')
    # log_string(args)
    
    # Buat folder untuk per block
    experiment_dir = 'log/sem_seg/' + args.log_dir
    filename = os.path.basename(str(args.point_cloud))
    filename = os.path.splitext(filename)[0]
    visual_dir = experiment_dir + '/visual/' + filename
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    NUM_CLASSES = args.num_classes
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = 'data/sem_seg_data/'

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    '''MODEL LOADING'''
    classifier = dgcnn_sem_seg(args).cuda()
    classifier = nn.DataParallel(classifier)
    checkpoint = torch.load(str(experiment_dir) + '/checkpoint/best_model.t7')
    classifier.load_state_dict(checkpoint)
    
    classifier = classifier.eval()
        
    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        
        # Edit num_batches = 1
        num_batches = 1 #len(TEST_DATASET_WHOLE_SCENE)   
        
        for batch_idx in range(num_batches):
            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))            
            scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            # batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9)) #XYZ, RGB, NormXYZ
            
            # Add Intensity
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 10)) #XYZ, RGB, I, NormXYZ
            
            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0 # Normalize color

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()
                torch_data = torch_data.transpose(2, 1)

                seg_pred  = classifier(torch_data)

                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1]
                pred_np = pred.detach().cpu().numpy()
                
                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                           pred_np[0:real_batch_size, ...],
                                           batch_smpw[0:real_batch_size, ...])
                
                
            pred_label =  np.argmax(vote_label_pool, 1)
            
            result_data = []
            
            # Change to UTM
            whole_scene_data[:, 0:2] += xy_min

            for i in range(whole_scene_label.shape[0]):
                data_point = [whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], whole_scene_data[i, 3], whole_scene_data[i, 4], whole_scene_data[i, 5], whole_scene_data[i, 6], pred_label[i]]
                result_data.append(data_point)
            
            result_data = np.array(result_data)
            result_data[result_data[:, -1] == 0, -1] = 6
            result_data[result_data[:, -1] == 1, -1] = 5
                       
            filename = os.path.join(visual_dir, filename + '-block-' + str(test_area))
            save_las(result_data, filename)
    
    # return result_data

if __name__ == '__main__':
        
    # Record the start time
    start_time = time.time()
    
    args = parse_args()
        
    data = read_las(args.point_cloud)
    
    xy_min = np.amin(data, axis=0)[0:2]
    data[:, 0:2] -= xy_min
    zeros_column = np.ones((data.shape[0], 1))
    data = np.hstack((data, zeros_column))
    
    data_dir = 'data/sem_seg_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    num_blocks_x, num_blocks_y = calculate_block_size(data, args.block_size)
    print("Number of blocks in x direction:", num_blocks_x)
    print("Number of blocks in y direction:", num_blocks_y)
    blocks = split_array(data, num_blocks_x, num_blocks_y, args.block_size)
    print("Total number of blocks generated:", len(blocks))   
    print("Split data into blocks")
    
    for i, block in enumerate(blocks):
        print("Block", i, "contains", len(block), "points")     
        block_data_bytes = block.tobytes()
        
        # Calculate the size of block_data in bytes
        block_data_size = len(block_data_bytes)
        
        # Check if the size is below 1 MB (1,048,576 bytes)
        if block_data_size > 1_048_576:
            filename_offground_npy = os.path.join(data_dir, f'Area_{i}.npy')
            # Save the block_data if it meets the size criteria
            np.save(filename_offground_npy, block)
            print(f'Saved: {filename_offground_npy} (size: {block_data_size} bytes)')
        else:
            print(f'Skipped: Block data size {block_data_size} bytes < 1 MB')

    # num_areas = sum(1 for file in os.listdir(data_dir) if file.endswith('.npy'))
    
    # Update: change how num_areas is called
    # List to store the numbers
    areas = []

    # Iterate over all files in the folder
    for filename in os.listdir(data_dir):
        # Check if the filename matches the pattern "Area_<number>.npy"
        match = re.match(r'Area_(\d+)\.npy', filename)
        if match:
            # Extract the number and convert it to an integer
            number = int(match.group(1))
            # Add the number to the list
            areas.append(number)

    # Sort the list of numbers
    areas.sort()
    
    # data = []
    
    print("Start classification")
    for test_area in areas:
        print("Classify Block ", test_area)
        args = parse_args(test_area)
        main(args)
        # result = main(args)
        # data.append(result)
    
    '''
    # Tambahan
    # Buat save las per block
    print("Classification done, save las file per blocks")
    experiment_dir = 'log/sem_seg/' + args.log_dir
    filename = os.path.basename(str(args.point_cloud))
    filename = os.path.splitext(filename)[0]
    
    visual_dir = experiment_dir + '/visual/' + filename
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)
            
    for i in range(len(data)):
        data[i][:, 0:2] += xy_min 
        filename_merge = os.path.join(visual_dir, filename + '-block-' + str(i))
        save_las(data[i], filename_merge)
    
    # Buat save las merge langsung jadi 1
    print("Classification done, merge all blocks")
    merged_data = np.concatenate(data, axis=0)
    merged_data[:, 0:2] += xy_min 
    
    save_las(merged_data, filename_merge)
    '''
        
    print("Done!")
        
    # Record the end time
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    shutil.rmtree(data_dir)

    print(f"Elapsed Time: {elapsed_time} seconds")