# 6. Define testing function

CUDA_LAUNCH_BLOCKING="1"

"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
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
from data_utils.check import check, filter

from data_utils.split_merge_las import *
from data_utils.dsm_dtm import create_dem

BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['0', '1']
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
    C = inFile.classification

    data = np.column_stack((x, y, z, r, g, b, C))
    
    xy_min = np.amin(data, axis=0)[0:2]
    data[:, 0:2] -= xy_min

    return data

def split(data, args):
    data_dir = 'data/sem_seg_data'
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    # Split Array:
    num_blocks_x, num_blocks_y = calculate_block_size(data, args.block_size)
    blocks = split_array(data, num_blocks_x, num_blocks_y, args.block_size)
        
    # Iterate through blocks and save each block into a separate .npy file
    for i, block_data in enumerate(blocks):
        filename_npy = os.path.join(data_dir, f'Area_{i}.npy')
        np.save(filename_npy, block_data)

def parse_args(test_area=None):
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=test_area, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--num_classes', type=int, default=2, help='How many classes used for segmentation [default: 9]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--point_cloud', type=str, required=True, help='Name of point cloud data')
    parser.add_argument('--block_size', type=int, default= 1000, help='Size of each block')
    parser.add_argument('--cell_size', type=float, default=1.0, help='Cell Size for creating DSM and DTM')
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def save_las(X, filename):
    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)
    las.x = X[:, 0]
    las.y = X[:, 1]
    las.z = X[:, 2]
    las.red = X[:, 3]
    las.green = X[:, 4]
    las.blue = X[:, 5]
    las.classification = X[:, 6]
    las.write(filename + ".las")

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    global visual_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # log_string('PARAMETER ...')
    # log_string(args)

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
        num_batches = len(TEST_DATASET_WHOLE_SCENE)
        
        for batch_idx in range(num_batches):
          
            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))            
            scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9)) #XYZ, RGB, NormXYZ
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
                batch_data[:, :, 3:6] /= 255 # Normalize color

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

            for i in range(whole_scene_label.shape[0]):
                data_point = [whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], whole_scene_data[i, 3], whole_scene_data[i, 4], whole_scene_data[i, 5], pred_label[i]]
                result_data.append(data_point)
            
            result_data = np.array(result_data)
    
    return result_data

def transform_utm(data, args):   
    # Add back the UTM Position
    filepath = args.point_cloud
    lasfile = laspy.read(filepath)
    x = lasfile.x
    y = lasfile.y
    z = lasfile.z

    data_label = np.column_stack((x, y, z))                
    xy_min = np.amin(data_label, axis=0)[0:2]
    data[:, 0:2] += xy_min
    
    global filename_pred
    
    filename = os.path.basename(str(args.point_cloud))
    filename = os.path.splitext(filename)[0]
    filename_pred = os.path.join(visual_dir, filename + '_pred')
    save_las(data, filename_pred)

if __name__ == '__main__':
        
    # Record the start time
    start_time = time.time()
    
    args = parse_args()
        
    data = read_las(args.point_cloud)
    
    print("Split data into blocks")
    
    split(data, args)
    
    data_folder = 'data/sem_seg_data'
    num_areas = sum(1 for file in os.listdir(data_folder) if file.endswith('.npy'))
    
    data = []
    
    print("Start classification")
    for test_area in range(num_areas):
        print("Classify Block ", test_area)
        args = parse_args(test_area)
        result = main(args)
        data.append(result)
    
    merged_data = np.concatenate(data, axis=0)

    print("Classification done, merge all blocks")

    transform_utm(merged_data, args)
    
    print("Classification Done!")
    
    # Output directory for DSM and DTM
    dtm = str(visual_dir) + "/dtm.tif"
    dsm = str(visual_dir) + "/dsm.tif"

    predict_result = filename_pred + '.las'
    
    if not check(predict_result):
        filter(predict_result, filename_pred)
    
    print("Creating DTM")
    create_dem(predict_result, dtm, args.cell_size, type='dtm')

    print("Creating DSM")
    create_dem(predict_result, dsm, args.cell_size, type='dsm')
        
    # Record the end time
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Elapsed Time: {elapsed_time} seconds")