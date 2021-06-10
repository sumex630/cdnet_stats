# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/5/9 13:37
@file: stats.py
@brief:
"""
import csv
import os
import subprocess

import pandas as pd

call = subprocess.call


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


def get_category_video(category_video):
    """
    csv 索引
    :param category_video:
    :return: dict
    """
    return {'category': category_video[0], 'video': category_video[1]}


def write_result_tocsv(stats_root, filename, data):
    """
    保存数据，依次追加
    :param stats_root:
    :param filename:
    :param data:
    :return:
    """
    if not os.path.exists(stats_root):
        os.mkdir(stats_root)
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


def compare_with_groungtruth(videoPath, binaryPath):
    """Compare your binaries with the groundtruth and return the confusion matrix"""
    statFilePath = os.path.join(videoPath, 'stats.txt')
    delete_if_exists(statFilePath)

    groundtruthPath = os.path.join(videoPath, 'groundtruth')
    retcode = call([os.path.join('exe', 'comparator.exe'),
                    videoPath, binaryPath],
                   shell=True)

    return read_cm_file(statFilePath)


def delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def read_cm_file(filePath):
    """Read the file, so we can compute stats for video, category and overall."""
    if not os.path.exists(filePath):
        print("The file " + filePath + " doesn't exist.\nIt means there was an error calling the comparator.")
        raise Exception('error')

    with open(filePath) as f:
        for line in f.readlines():
            if line.startswith('cm:'):
                numbers = line.split()[1:]
                return [int(nb) for nb in numbers[:5]]


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


def stats(dataset_root, binary_root, stats_root):
    """
    CDNET 数据集度量统计
    :param dataset_root: 数据集路径
    :param binary_root: 检测结果路径
    :param stats_root: 保存统计结果路径
    :return:
    """
    save_filename = ''

    for dirpath, dirnames, filenames in os.walk(binary_root):
        if filenames:  #  and 'boats' in dirpath  corridor traffic
            # print('正在计算评估指标：', dirpath)
            dirpath_list = dirpath.replace('\\', '/').split('/')  # 切割路径
            algorithm_name_index = dirpath_list.index(os.path.basename(binary_root))  # binary_root 文件位置
            algorithm_name_type = dirpath_list[algorithm_name_index:-2]  #
            save_filename = '_'.join(algorithm_name_type)  # 保存stats时的文件名
            save_filename = save_filename + '.csv'
            category_video = dirpath_list[-2:]  # 保存时的索引名称 stats

            # 数据集的视频序列路径
            dataset_video_path = os.path.join(dataset_root, dirpath_list[-2], dirpath_list[-1])
            if not os.path.exists(dataset_video_path):
                dataset_video_path = os.path.join(dataset_root, dirpath_list[-2], dirpath_list[-2], dirpath_list[-1])

            if is_valid_video_folder(dataset_video_path):
                # 计算混淆矩阵
                confusion_matrix = compare_with_groungtruth(dataset_video_path, dirpath)

                frames_stats = get_stats(confusion_matrix)  # 7中度量
                frames_category = get_category_video(category_video)  # 索引名 stats
                frames_category.update(frames_stats)  # 合并数据

                # 保存数据
                write_result_tocsv(stats_root, save_filename, frames_category)

    if save_filename:
        write_category_and_overall_tocsv(stats_root, save_filename)


if __name__ == '__main__':
    # dataset_root = 'F:/Dataset/CDNet2012/'  # 数据集根目录
    # 数据集根目录
    dataset_root = 'F:\Dataset\CDNet2014\dataset'
    # 检测结果根目录
    binary_root = 'F:/Pycharm/01_CV/20210519_idea/20210526_yolact/yolact/results/yolact_diff2014/yolact_diff2014'
    # 统计结果根目录
    stats_root = './results_stats'

    stats(dataset_root, binary_root, stats_root)
