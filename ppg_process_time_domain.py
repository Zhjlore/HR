from statistics import mean

import numpy as np
import os
import matplotlib.pyplot as plt
from rawdata_analysis import rawdata_parse
import sensor_quality_model as sensor_quality
from scipy.signal import butter, filtfilt, detrend
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


def baseline_removal_ppg(signal, window_size):
    """
    基于滑动窗口平均的PPG数据去基线
    signal (numpy.array): 输入的PPG信号
    window_size (int): 窗口大小
    """
    baseline_removed_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        start_index = max(0, i - window_size // 2)  # 当前索引减去窗口大小的一半
        end_index = min(len(signal), i + window_size // 2 + 1)  # 当前索引加上窗口大小的一半
        baseline = np.mean(signal[start_index:end_index])
        baseline_removed_signal[i] = signal[i] - baseline

    return baseline_removed_signal


def gaussian_smoothing(signal, sigma):
    """
    高斯滤波函数
    signal (numpy.array): 输入信号
    sigma (float): 高斯核的标准差
    """
    smoothed_signal = gaussian_filter(signal, sigma=sigma)
    return smoothed_signal


def peak_detection(ppg_signal, neighbor):
    peaks = []
    for i in range(len(ppg_signal)):
        # 检查是否在边界内
        if i < neighbor or i >= len(ppg_signal) - neighbor:
            continue
        # 判断是否为峰值点
        if ppg_signal[i] > max(ppg_signal[i-neighbor:i]) and ppg_signal[i] > max(ppg_signal[i+1:i+neighbor+1]):
            peaks.append(i)
    return peaks


# def peak_detection(ppg_signal, neighbor):
#     # 预先计算每个点的邻近窗口内的最大值
#     max_neighbors = [max(ppg_signal[i-neighbor:i+neighbor+1]) for i in range(len(ppg_signal))]
#
#     peaks = []
#     for i in range(len(ppg_signal)):
#         # 检查是否在边界内，如果是，则跳过
#         if i < neighbor or i >= len(ppg_signal) - neighbor:
#             continue
#         # 判断是否为峰值点
#         if ppg_signal[i] == max_neighbors[i]:
#             peaks.append(i)
#     return peaks

    # 计算前10秒的心率
    # segment = ppg_signal[:fs * window_size]
    # ppg_peaks = peak_detection(segment, 5)
    # peak_indices = list(ppg_peaks)
    # rr_intervals = []
    # for j in range(len(peak_indices) - 1):
    #     rr_interval = (peak_indices[j + 1] - peak_indices[j])  # 计算RR间期时不应先算时间，而应该算数据点
    #     rr_intervals.append(rr_interval)  # 获取所有的RR间期
    # if rr_intervals:
    #     avg_rr_interval = np.mean(rr_intervals)
    #     heart_rate = 60 / (avg_rr_interval / fs)
    #     heart_rates.append(heart_rate)
    #     # print("heart_rates", heart_rates)
    # # i += window_size*fs  # 下一段数据的起点i应与前面间隔10s数据
    # i += fs


def calculate_heart_rates(ppg_signal, fs, window_size=10):
    heart_rates = []
    i = 0  # i为信号数据点位置
    # 计算心率，每隔1秒计算一次
    while i < len(ppg_signal):
        segment = ppg_signal[i:i + fs * window_size]
        # print("i", i)
        # print("segment", len(segment))
        if len(segment) > 225:
            ppg_peaks = peak_detection(segment, 5)
            peak_indices = list(ppg_peaks)
            rr_intervals = []
            for j in range(len(peak_indices) - 1):
                # 计算RR间期时不应先算时间，而应该算数据点
                rr_interval = (peak_indices[j + 1] - peak_indices[j])
                rr_intervals.append(rr_interval)
            if rr_intervals:
                avg_rr_interval = np.mean(rr_intervals)
                heart_rate = int(60 / (avg_rr_interval / fs))
                heart_rates.append(heart_rate)
        i += fs
    return heart_rates


def calculate_error(reference_signal, test_signal):
    # 对齐信号长度
    min_length = min(len(reference_signal), len(test_signal))
    reference_signal = reference_signal[:min_length]
    test_signal = test_signal[:min_length]

    # 计算误差
    error = np.array(reference_signal) - np.array(test_signal)
    return error


if __name__ == '__main__':

    n = 3068
    fs = 25
    # 静止

    file_dir = r'D:\ZHJ\data_compare\06_static\3_2024_04_23_10_07_47'
    file_name = r'HeartRate_Static_zhj_20240423_100423_67d7.rawdata'

    file_path = file_dir + os.sep + file_name

    rawdata_json, sample_info = rawdata_parse(file_path)

    ppg_g_sensors = list()   # PPG传感器数据
    alg_heartrate = list()   # 手表心率
    reference_heartrate = list()  # 心率带心率
    ppg_freq_list = []
    for data in rawdata_json:
        if 'sensor_data' in data:
            sensor_data = data['sensor_data']
            # print(data['timestamp'],data['unix_timestamp'])
            if 'PPG-G' in sensor_data:
                # print('PPG-G', len(sensor_data['PPG-G']))
                ppg_freq_list.append(len(sensor_data['PPG-G']))
                ppg_g_sensors = np.hstack((ppg_g_sensors, sensor_data['PPG-G']))

        if 'alg_log' in data:
            sensor_info = data['alg_log']

            reference_heartrate.append(sensor_info['reference_heartrate'])
            alg_heartrate.append(sensor_info['heartrate'])

    ppg_g_sensors = (np.array(ppg_g_sensors) - 1000000)

    # 1 预处理 去除基线
    baseline_removed_ppg = baseline_removal_ppg(ppg_g_sensors, 10)

    # 2 平滑去噪  心跳信号频率[0.5-3.6Hz]
    sigma = 5.0  # 高斯核的标准差
    smoothed_ppg = gaussian_smoothing(baseline_removed_ppg, sigma)
    ppg_signal = smoothed_ppg
    print("num of ppg_signal:", len(ppg_signal))
    print("time of ppg_signal", len(ppg_signal)/25, "s")

    # 获取波峰位置索引
    ppg_peaks = peak_detection(ppg_signal[1:n], 5)
    print("ppg_peaks", ppg_peaks)

    # rr_list = []
    # for i in range(0, len(ppg_peaks)-1, 1):
    #     rr_list.append(ppg_peaks[i+1] - ppg_peaks[i])
    #
    # print("rr_list", rr_list)
    # sum_rr = sum(rr_list)
    # rr_ave = sum_rr / len(rr_list)
    # my_hr_time_domain = (60 / ((sum(rr_list)) / len(rr_list) / 25))
    # print("my_hr_time_domain", my_hr_time_domain)

    # 计算心率
    # heart_rates = calculate_heart_rate(ppg_signal, 25, 10)

    heart_rates = calculate_heart_rates(ppg_signal[1:n], 25, window_size=10)
    print("heart_rates", heart_rates)
    print("num of heart_rates", len(heart_rates))

    # 绘制心率带曲线与计算得到心率曲线
    window_size = 10

    # plt.figure("heart_rates")
    # plt.plot(reference_heartrate[5:], color='r', label='reference_heartrate')
    # plt.plot(heart_rates, color='g', label='heart_rates')
    # plt.ylim(60, 200)
    # plt.legend(loc='upper center')

    # 计算误差
    # error_value = calculate_error(reference_heartrate, heart_rates)
    # print("error:", error_value)
    print("reference_heartrate", reference_heartrate)
    # print("len_reference_heartrate",len(reference_heartrate))
    #
    # # 原始PPG数据波形
    # time = np.arange(len(ppg_signal[:n])) / fs
    # # time = np.arange(len(ppg_signal[:n]))
    # plt.figure('ppg_g_sensors', figsize=(8, 6))
    # plt.plot(time, ppg_g_sensors[:n], color='r')
    # peaks = peak_detection(ppg_signal[:n], 5)
    # peaks_array = np.array(peaks)
    # plt.scatter(peaks_array / fs, ppg_g_sensors[peaks], color='g', label='Peaks')
    # plt.xlabel("number of samples")
    # plt.ylabel("amplitude")
    #
    # # 去基线后PPG数据波形
    # plt.figure('ppg after baseline removal', figsize=(8, 6))
    # plt.plot(time, baseline_removed_ppg[:n], color='r')
    # peaks = peak_detection(ppg_signal[:n], 5)
    # peaks_array = np.array(peaks)
    # plt.scatter(peaks_array / fs, baseline_removed_ppg[peaks], color='g', label='Peaks')
    # plt.xlabel("number of samples")
    # plt.ylabel("amplitude")
    #
    # # 平滑去噪后波形
    # plt.figure('smoothed_ppg', figsize=(8, 6))
    # plt.plot(time, smoothed_ppg[:n], color='r')
    # peaks = peak_detection(ppg_signal[:n], 5)
    # peaks_array = np.array(peaks)
    # plt.scatter(peaks_array / fs, smoothed_ppg[peaks], color='g', label='Peaks')
    # plt.xlabel("number of samples")
    # plt.ylabel("amplitude")
    #
    # # 绘制波峰位置曲线
    # # time = np.arange(len(ppg_signal[:n])) / fs
    # plt.figure(figsize=(8, 6))
    # plt.plot(time, ppg_signal[:n], label='PPG Signal')
    # peaks = peak_detection(ppg_signal[:n], 5)
    # peaks_array = np.array(peaks)
    # plt.scatter(peaks_array / fs, ppg_signal[peaks], color='red', label='Peaks')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('PPG Signal with Peaks')
    # plt.legend()

    plt.show()


