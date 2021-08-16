# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/5/9 13:37
@file: stats.py
@brief:
"""
import cv2
import pandas as pd
import argparse
import csv
import os
import subprocess

import torch
from tqdm import tqdm

call = subprocess.call

parser = argparse.ArgumentParser()

# Input arguments  dataset_root, binary_root, stats_root
parser.add_argument("--dr", help="filepath to the dataset_root", default="/home/lthpc/sumex/datasets/cdnet2014/dataset")
parser.add_argument("--br", help="filepath to the binary_root", default="")
parser.add_argument("--sr", help="filepath to the output_path", default="results_stats")
parser.add_argument("--sub", help="filepath to the sub_output_path", default="")

args = parser.parse_args()


class ConfusionMatrix:

    def __init__(self):

        self.TP = 0  # 真正
        self.FP = 0  # 假正
        self.FN = 0  # 假负
        self.TN = 0  # 真负

        self.BG = 0
        self.FG = 255

        self.ones = None
        self.zeros = None

    def evaluate(self, mask, groundtruth, roi):
        self.ones = torch.ones_like(mask, dtype = torch.float)
        self.zeros = torch.zeros_like(mask, dtype = torch.float)

        TP_mask = torch.where((mask == self.FG) & (groundtruth == self.FG) & (roi == self.FG), self.ones, self.zeros)
        FP_mask = torch.where((mask == self.FG) & (groundtruth == self.BG) & (roi == self.FG), self.ones, self.zeros)
        FN_mask = torch.where((mask == self.BG) & (groundtruth == self.FG) & (roi == self.FG), self.ones, self.zeros)
        TN_mask = torch.where((mask == self.BG) & (groundtruth == self.BG) & (roi == self.FG), self.ones, self.zeros)

        self.TP += torch.sum(TP_mask)
        self.FP += torch.sum(FP_mask)
        self.FN += torch.sum(FN_mask)
        self.TN += torch.sum(TN_mask)


def get_stats(cm):
    """
    Return the usual stats for a confusion matrix
    • TP (True Positive)：表示正确分类为前景的像素个数。
    • TN (True Negative)：表示正确分类为背景的像素个数。
    • FP (False Positive)：表示错误分类为前景的像素个数。
    • FN (False Negative)：表示错误分类为背景的像素个数。
    1. Recall 即召回率，表示算法正确检测出的前景像素个数占基准结果图像中所有前景像素个数的百分比，数值在 0 到 1 之间，结果越趋近于 1 则说明算法检测效果越好
    2. Precision 即准确率，表示算法正确检测出的前景像素个数占所有检测出的前景像素个数的百分比，数值在 0 到 1 之间，结果越趋近于 1 则说明算法检测效果越好
    3. F-Measure (F1-Score) 就是这样一个指标，常用来衡量二分类模型精确度，它同时兼顾了分类模型的 Recall 和 Precision，是两个指标的一种加权平均，
        最小值是 0，最大值是 1，越趋近于 1 则说明算法效果越好
    4. Specificity 表示算法正确检测出的背景像素个数占基准结果图像中所有背景像素个数的百分比，数值在 0 到 1 之间，越趋近于 1 则说明算法检测效果越好
    5. FPR 表示背景像素被错误标记为前景的比例，数值在 0 到 1 之间，和上述四个指标相反，该值越趋近于 0，则说明算法检测效果越好
    6. FNR 表示前景像素被错误标记为背景的比例，数值在 0 到 1 之间，同样该值越趋近于 0，则说明算法检测效果越好
    7. PWC 表示错误率，包括前景像素和背景像素，数值在 0 到 ？ 之间，该值越趋近于 0，则说明算法检测效果越好
    :param cm: 混淆矩阵
    :return:
    """
    # TP = cm[0, 0]
    # FN = cm[0, 1]
    # FP = cm[1, 0]
    # TN = cm[1, 1]
    # TP, FN, FP, TN, SE = cm
    TP, FP, FN, TN, SE = cm

    recall = TP / (TP + FN)
    specficity = TN / (TN + FP)
    fpr = FP / (FP + TN)
    fnr = FN / (TP + FN)
    pbc = 100.0 * (FN + FP) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    fmeasure = 2.0 * (recall * precision) / (recall + precision)

    stats_dic = {'Recall': recall,
                 'Precision': precision,
                 'Specificity': specficity,
                 'FPR': fpr,
                 'FNR': fnr,
                 'PWC': pbc,
                 'FMeasure': fmeasure,
                 }

    return stats_dic


def get_category_video(category, video):
    """
    csv 索引
    :param category_video:
    :return: dict
    """
    return {'category': category, 'video': video}


def write_result_tocsv(stats_root, filename, data):
    """
    保存数据，依次追加
    :param stats_root:
    :param filename:
    :param data:
    :return:
    """
    if not os.path.exists(stats_root):
        os.makedirs(stats_root)
    stats_path = os.path.join(stats_root, filename)
    is_stats = True
    try:
        pd.read_csv(stats_path)
    except Exception as e:
        is_stats = False

    header = list(data.keys())
    with open(stats_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        if not is_stats:  # 第一次写入将创建 header
            writer.writeheader()  # 写入列名
        writer.writerows([data])  # 写入数据


def is_valid_video_folder(path):
    """
    视频文件夹是否有效
    A valid video folder must have \\groundtruth, \\input, ROI.bmp, temporalROI.txt
    :param path:
    :return:
    """
    return os.path.exists(os.path.join(path, 'groundtruth')) and \
           os.path.exists(os.path.join(path, 'input')) and \
           os.path.exists(os.path.join(path, 'ROI.bmp')) and \
           os.path.exists(os.path.join(path, 'temporalROI.txt'))


def get_temporalROI(path):
    """
    获取检测帧的范围
    :param path:dataset/baseline/baseline/highway
    :return:['470', '1700'] [起始帧，结束帧]
    """
    path = os.path.join(path, 'temporalROI.txt')
    with open(path, 'r') as f:
        avail_frames = f.read()

    return avail_frames.split(' ')


def get_roi(video_path):
    roi_path = os.path.join(video_path, "ROI.bmp")
    roi = cv2.imread(roi_path, 0)
    if "traffic" in video_path:
        # 数据集中 traffic 视频序列 的ROI.bmp 与数据集尺寸不同
        roi_path_jpg = os.path.join(video_path, "ROI.jpg")
        roi_size = cv2.imread(roi_path_jpg, 0).shape
        roi = cv2.resize(cv2.imread(roi_path, 0), (roi_size[1], roi_size[0]))

    return roi


def get_gt(video_path, filename):
    gt_path = os.path.join(video_path, "groundtruth", filename.replace("bin", "gt").replace("jpg", "png"))

    return cv2.imread(gt_path, 0)


def compare_with_groundtruth(CM, videoPath, binaryPath):
    """Compare your binaries with the groundtruth and return the confusion matrix"""
    # print("videoPath", videoPath)
    # print("binaryPath", binaryPath)
    roi = get_roi(videoPath)

    vaild_frames = get_temporalROI(videoPath)  # 有效帧范围
    start_frame_id = int(vaild_frames[0])  # 起始帧号
    end_frame_id = int(vaild_frames[1])  # 结束帧号
    # print(vaild_frames)

    for bin_filename in os.listdir(binaryPath)[start_frame_id - 1:end_frame_id + 1]:
        bin_path = os.path.join(binaryPath, bin_filename)

        CM.evaluate(torch.from_numpy(cv2.imread(bin_path, 0)),
                    torch.from_numpy(get_gt(videoPath, bin_filename)),
                    torch.from_numpy(roi))

    return [CM.TP.numpy(), CM.FP.numpy(), CM.FN.numpy(), CM.TN.numpy(), 0]


def write_category_and_overall_tocsv(stats_root, save_filename):
    """
    保存类统计
    :param stats_root:
    :param save_filename:
    :return:
    """
    stats_path = os.path.join(stats_root, save_filename)
    stats_data = pd.read_csv(stats_path)
    category_col = stats_data['category']
    # 去重，不改变顺序
    categories = []
    for cat in category_col:
        if cat not in categories:
            categories.append(cat)

    overall = []
    write_result_tocsv(stats_root, save_filename, {})  # 空三行
    write_result_tocsv(stats_root, save_filename, {})
    write_result_tocsv(stats_root, save_filename, {})

    # category
    for category in categories:
        category_dict = {'category': category, 'video': ''}
        category_stats = stats_data[stats_data['category'] == category]

        category_dict['Recall'] = category_stats['Recall'].mean()
        category_dict['Precision'] = category_stats['Precision'].mean()
        category_dict['Specificity'] = category_stats['Specificity'].mean()
        category_dict['FPR'] = category_stats['FPR'].mean()
        category_dict['FNR'] = category_stats['FNR'].mean()
        category_dict['PWC'] = category_stats['PWC'].mean()
        category_dict['FMeasure'] = category_stats['FMeasure'].mean()

        overall.append(category_dict)
        write_result_tocsv(stats_root, save_filename, category_dict)

    write_result_tocsv(stats_root, save_filename, {})

    # overall
    overall_stats = pd.DataFrame(overall)
    category_dict = {'category': 'overall', 'video': ''}

    category_dict['Recall'] = overall_stats['Recall'].mean()
    category_dict['Precision'] = overall_stats['Precision'].mean()
    category_dict['Specificity'] = overall_stats['Specificity'].mean()
    category_dict['FPR'] = overall_stats['FPR'].mean()
    category_dict['FNR'] = overall_stats['FNR'].mean()
    category_dict['PWC'] = overall_stats['PWC'].mean()
    category_dict['FMeasure'] = overall_stats['FMeasure'].mean()

    write_result_tocsv(stats_root, save_filename, category_dict)


def get_save_filename(bin_path):
    """
    获取保存数据时的文件名
    """
    binary_root_list = bin_path.replace('\\', '/').split('/')  # 切割路径
    save_filename_list = binary_root_list[-2:]  # 最后两个路径名称联合

    # 保存stats时的文件名
    return '_'.join(save_filename_list) + '.csv'


def get_categories(dataset_dir):
    """
    Stores the list of categories as string and the videos of each
    category in a dictionary.
    """
    categories = sorted(os.listdir(dataset_dir), key=lambda v: v.upper())

    videos = dict()

    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        videos[category] = sorted(os.listdir(category_dir), key=lambda v: v.upper())

    return categories, videos


def stats(dataset_root, binary_root, stats_root):
    """
    CDNET 数据集度量统计
    :param dataset_root: 数据集路径
    :param binary_root: 检测结果路径
    :param stats_root: 保存统计结果路径
    :return:
    """
    ii = 0
    # 保存数据时的文件名称
    save_filename = get_save_filename(binary_root)
    # Get the names of the categories and the videos
    categories, videos = get_categories(dataset_root)

    #  os.walk(binary_root) 在ubuntu 中遍历出的文件名乱序
    p_bar = tqdm(total = 53)
    # Loop over all categories that were retrieved
    for category in categories:
        # Loop over all videos
        for video in videos[category]:
            # Definition of the video directory path
            video_dir_datasets = os.path.join(dataset_root, category, video)
            video_dir_binary = os.path.join(binary_root, category, video)

            if not os.path.exists(video_dir_binary):
                continue

            # ii += 1
            # print('{} 正在计算评估指标：{}'.format(ii, video_dir_datasets))

            if is_valid_video_folder(video_dir_datasets):  # 检测数据集是否有效
                # 混淆矩阵
                CM = ConfusionMatrix()
                confusion_matrix = compare_with_groundtruth(CM, video_dir_datasets, video_dir_binary)

                frames_stats = get_stats(confusion_matrix)  # 7中度量
                frames_category = get_category_video(category, video)  # 索引名 stats
                frames_category.update(frames_stats)  # 合并数据

                # 保存数据
                write_result_tocsv(stats_root, save_filename, frames_category)

            p_bar.update(1)

    write_category_and_overall_tocsv(stats_root, save_filename)
    p_bar.close()


if __name__ == '__main__':
    # 数据集根目录
    dataset_root = 'E:/00_Datasets/dataset2014/dataset'
    # 检测结果根目录
    # binary_root = r"E:\01_PyCharm\01_CV\20210714_instance_segmentation\20210716_yolact\results\202108015_yolact_vibe_mask\mt=45_ratio=0.3_yolact_vibe20210815"
    binary_root = r"E:\00_Datasets\subsense\subsense_cdnet2014\results"
    # 统计结果根目录
    stats_root = r'results_stats'
    # #################################################################
    # # python python_stats/stats.py --br  --sr
    # dataset_root = args.dr
    # # "/home/lthpc/sumex/20210807_ISBS/20210811_ISBS/results/yolact_vibe/20210812_ratio=0.3_opt=0"
    # binary_root = args.br
    # # "/home/lthpc/sumex/20210807_ISBS/20210811_ISBS/results_stats"
    # stats_root = os.path.join(args.sr, args.sub) if args.sub else args.sr

    stats(dataset_root, binary_root, stats_root)
