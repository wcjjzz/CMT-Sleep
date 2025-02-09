'''
除了原有的预处理signal_extract外
还加入了针对目前遇到的数据集的不同预处理接口
Sleep-EDF, DREAMS, Figshare, hospital
'''



# 下载mne库
# !pip install mne



# 导入库
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import mne
from mne.datasets.sleep_physionet.age import fetch_data
# from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import h5py
import argparse
import pyedflib
import pandas as pd
from datetime import datetime, timedelta



# 用于接收控制台提供的参数
def parse_option():
    parser = argparse.ArgumentParser('Argument for data generation')
    parser.add_argument('--save_path', type=str, default='./extract_dataset_single_epoch',
                        help='Path to store project results')

    opt = parser.parse_args()
    return opt



# 针对Sleep_EDF数据集，需要输入<受试者编号数组subjects[]>和<天数数组days[]>
# 返回PSG信号数组，标签二维数组，PSG信号长度数组，均值数组，标准差数组
def signal_extract(subjects, days, channel='eeg1', filter=True, freq=[0.2, 40]):
    # 需要忽略的数据
    ignore_data = [[13, 2], [36, 1], [39, 1], [39, 2], [52, 1], [68, 1], [68, 2], [69, 1], [69, 2], [78, 1], [78, 2],
                   [79, 1], [79, 2]]
    # 所有通道
    all_channels = ('EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    # 标记是否是第一份数据
    first_sub_flag = 0

    # 遍历所有数据并保存
    for sub in subjects:
        for day_ in days:
            if [sub, day_] in ignore_data:
                continue
            [data] = fetch_data(subjects=[sub], recording=[day_])
            signal2idx = {"eeg1": 0, "eeg2": 1, "eog": 2, "emg": 3}

            all_channels_list = list(all_channels)
            all_channels_list.remove(all_channels[signal2idx[channel]])
            exclude_channels = tuple(all_channels_list)

            sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)

            annot = mne.read_annotations(data[1])

            # 睡眠阶段和数字的对应关系
            ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                         "Sleep stage 4": 4, "Sleep stage R": 5}
            #     # "Sleep stage ?": 5,
            #     # "Movement time": 5

            # 切割
            annot.crop(annot[1]['onset'] - 30 * 60,
                       annot[-2]['onset'] + 30 * 60)

            sleep_signals.set_annotations(annot, emit_warning=False)

            events, _ = mne.events_from_annotations(
                sleep_signals, event_id=ann2label, chunk_duration=30.)

            # Filtering
            tmax = 30. - 1. / sleep_signals.info['sfreq']

            if filter == True:
                sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

            # Breaking into Epochs
            epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                                     event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                                     on_missing='warn')

            sig_epochs = []
            label_epochs = []

            mean_epochs = []
            std_epochs = []

            signal_mean = np.mean(np.array([epochs_data]))
            signal_std = np.std(np.array([epochs_data]))

            for ep in range(len(epochs_data)):
                for sig in epochs_data[ep]:
                    sig_epochs.append(sig)

                sleep_stage = epochs_data[ep].event_id

                if sleep_stage == {"Sleep stage W": 0}:
                    label_epochs.append(0)
                if sleep_stage == {"Sleep stage 1": 1}:
                    label_epochs.append(1)
                if sleep_stage == {"Sleep stage 2": 2}:
                    label_epochs.append(2)
                if sleep_stage == {"Sleep stage 3": 3}:
                    label_epochs.append(3)
                if sleep_stage == {"Sleep stage 4": 4}:
                    label_epochs.append(3)
                if sleep_stage == {"Sleep stage R": 5}:
                    label_epochs.append(4)

                mean_epochs.append(signal_mean)
                std_epochs.append(signal_std)

            sig_epochs = np.array(sig_epochs)
            mean_epochs = np.array(mean_epochs)
            std_epochs = np.array(std_epochs)
            label_epochs = np.array(label_epochs)

            if first_sub_flag == 0:
                main_ext_raw_data = sig_epochs
                main_labels = label_epochs
                main_sub_len = np.array([len(epochs_data)])
                main_mean = mean_epochs
                main_std = std_epochs
                first_sub_flag = 1
            else:
                main_ext_raw_data = np.concatenate((main_ext_raw_data, sig_epochs), axis=0)
                main_labels = np.concatenate((main_labels, label_epochs), axis=0)
                main_sub_len = np.concatenate((main_sub_len, np.array([len(epochs_data)])), axis=0)
                main_mean = np.concatenate((main_mean, mean_epochs), axis=0)
                main_std = np.concatenate((main_std, std_epochs), axis=0)

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std



# 针对Sleep_EDF数据集，需要输入<psg信号文件地址psg_file>和<注释文件地址annotation_file>
# 返回psg信号，标签数组，psg信号长度，均值，标准差
# 这个数据预处理切割时间从熄灯到倒数第二个注释
def Sleep_EDF_SC_signal_extract(psg_file, annotation_file, channel='eeg1', filter=True, freq=[0.2, 40]):
    # 所有通道
    all_channels = (
        'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    data = [psg_file, annotation_file]
    signal2idx = {"eeg1": 0, "eeg2": 1, "eog": 2, "emg": 3}

    all_channels_list = list(all_channels)
    all_channels_list.remove(all_channels[signal2idx[channel]])
    exclude_channels = tuple(all_channels_list)

    sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)

    annot = mne.read_annotations(data[1])

    ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                 "Sleep stage 4": 4, "Sleep stage R": 5}
    #     # "Sleep stage ?": 5,
    #     # "Movement time": 5

    #########################
    # print("------------------------------", annot[1], "---------------------------------")
    # data_sleep_light_of_time = pd.read_excel("D:\hkk\项目_可解释性睡眠分期\Transformer\原始数据集\sleep-edf-database-expanded-1.0.0\SC-subjects.xlsx")
    # T_Light_off = data_sleep_light_of_time[['LightsOff']]
    #########################

    sc4_index = annotation_file.find("SC4")
    # 从"SC4"的位置开始，提取后面三个字符
    if sc4_index != -1:
        extracted_chars = annotation_file[sc4_index + 3:sc4_index + 6]
        print(extracted_chars)

    # 获取到受试者及天数信息，通过light_off_time函数获取熄灯时间
    sub = int(extracted_chars[:2])
    day = int(extracted_chars[2])
    start_to_light_off = light_off_time(sub, day, annotation_file)
    # print(sub)
    # print(day)
    # print("annot[1][oneset] = ", annot[1]['onset'])

    # 按熄灯时间切割
    # 目前受试者的起床时间还没有确定，所以采用倒数第二个event的时间
    annot.crop(annot[0]['onset'] + start_to_light_off,
               annot[-2]['onset'])

    sleep_signals.set_annotations(annot, emit_warning=False)

    events, _ = mne.events_from_annotations(
        sleep_signals, event_id=ann2label, chunk_duration=30.)

    # Filtering
    tmax = 30. - 1. / sleep_signals.info['sfreq']

    if filter == True:
        sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

    # Breaking into Epochs
    epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                             event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                             on_missing='warn')

    sig_epochs = []
    label_epochs = []

    mean_epochs = []
    std_epochs = []

    signal_mean = np.mean(np.array([epochs_data]))
    signal_std = np.std(np.array([epochs_data]))

    for ep in range(len(epochs_data)):
        for sig in epochs_data[ep]:
            sig_epochs.append(sig)

        sleep_stage = epochs_data[ep].event_id

        if sleep_stage == {"Sleep stage W": 0}:
            label_epochs.append(0)
        if sleep_stage == {"Sleep stage 1": 1}:
            label_epochs.append(1)
        if sleep_stage == {"Sleep stage 2": 2}:
            label_epochs.append(2)
        if sleep_stage == {"Sleep stage 3": 3}:
            label_epochs.append(3)
        if sleep_stage == {"Sleep stage 4": 4}:
            label_epochs.append(3)
        if sleep_stage == {"Sleep stage R": 5}:
            label_epochs.append(4)

        mean_epochs.append(signal_mean)
        std_epochs.append(signal_std)

    sig_epochs = np.array(sig_epochs)
    mean_epochs = np.array(mean_epochs)
    std_epochs = np.array(std_epochs)
    label_epochs = np.array(label_epochs)

    main_ext_raw_data = sig_epochs
    main_labels = label_epochs
    main_sub_len = np.array([len(epochs_data)])
    main_mean = mean_epochs
    main_std = std_epochs

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std
# 输入<受试者编号sub>,<天数day>和<注释文件地址edf_file_path>
# 返回熄灯时间和开始记录时间的差值
def light_off_time(sub, day, edf_file_path):
    # 示例数据
    # sub = 0
    # day = 1
    # edf_file_path = "D:\\hkk\\项目_可解释性睡眠分期\\Transformer\\原始数据集\\sleep-edf-database-expanded-1.0.0\\sleep-cassette\\SC4001EC-Hypnogram.edf"

    # 记录sleep-cassette受试者熄灯时间的excel文件--SC-subjects.xlsx(更改为自己的地址即可)
    excel_path = "D:\\hkk\\项目_可解释性睡眠分期\\Transformer\\原始数据集\\sleep-edf-database-expanded-1.0.0\\SC-subjects.xlsx"

    # 打开EDF文件
    f = pyedflib.EdfReader(edf_file_path)

    # 开始记录PSG的时间
    start_time = f.getStartdatetime()
    # print("start_record_time:", start_time)

    # 受试者熄灯时间
    data_sleep_light_of_time = pd.read_excel(excel_path)
    filtered_data = data_sleep_light_of_time[(data_sleep_light_of_time['subject'] == sub) & (data_sleep_light_of_time['night'] == day)]
    # 确保 lights_off_time 是字符串
    lights_off_time_str = filtered_data['LightsOff'].iloc[0]
    # print("lights_of_time:",lights_off_time_str)

    # 将字符串转换为 datetime.time 对象
    # lights_off_time_obj = datetime.strptime(lights_off_time_str, '%H:%M:%S').time()

    # 创建完整的日期时间对象
    lights_off_datetime = datetime.combine(start_time.date(), lights_off_time_str)

    # 如果 lights_off_time 在第二天，需要调整日期
    if lights_off_time_str < start_time.time():
        lights_off_datetime += timedelta(days=1)

    # 计算时间差
    time_difference = lights_off_datetime - start_time

    # print("time_diffence(h):",time_difference)
    # print("time_diffence(s):",time_difference.total_seconds())

    f.close()

    return time_difference.total_seconds()



# 针对Dreams数据集，需要输入<PSG信号地址psg_path>和<注释文件地址hypnogram_path>
# 返回psg信号，标签数组，psg信号长度，均值，标准差
# 需要注意原本的注释文件数字对应的睡眠阶段和Sleep-edf是不一样的，但是我们事先做了注释转换，所以此处代码的对应关系不需要改变
def DREAMS_signal_extract(psg_path, hypnogram_path, channel='FP1-A2', filter=True, freq=[0.2, 40]):
    id_idx1 = psg_path.find("subject")
    id_idx2 = psg_path.find(".edf")
    id = psg_path[id_idx1+7:id_idx2]
    id = int(id)
    if id != 2 and id != 6 and id != 9:
        all_channels = ('ECG', 'FP1-A2', 'CZ-A1', 'EMG1', 'EOG1', 'VTH', 'VAB',
                        'NAF2P-A1', 'NAF1', 'PHONO', 'PR', 'SAO2', 'PCPAP', 'POS',
                        'EOG2', 'O1-A2', 'FP2-A1', 'O2-A1', 'CZ2-A1', 'EMG2',
                        'PULSE', 'VTOT', 'EMG3')
    else:
        # DREAMS数据集2,6,9受试者专属
        all_channels = ('ECG', 'FP1-A2', 'CZ-A1', 'EMG1', 'EOG1-A2', 'VTH', 'VAB',
                        'NAF2P-A1', 'NAF1', 'PHONO', 'PR', 'SAO2', 'PCPAP', 'POS',
                        'EOG2-A2', 'O1-A2', 'FP2-A1', 'O2-A1', 'CZ2-A1', 'EMG2',
                        'PULSE', 'VTOT', 'EMG3')


    data = [psg_path, hypnogram_path]
    print("preparing: " + data[0] + " " + data[1])
    if id != 2 and id != 6 and id != 9:
        signal2idx = {"ECG": 0, "FP1-A2": 1, "CZ-A1": 2, "EMG1": 3, "EOG1" : 4}
    else:
        signal2idx = {"ECG": 0, "FP1-A2": 1, "CZ-A1": 2, "EMG1": 3, "EOG1-A2": 4}

    all_channels_list = list(all_channels)
    all_channels_list.remove(all_channels[signal2idx[channel]])
    exclude_channels = tuple(all_channels_list)

    # print(exclude_channels)
    # print(len(exclude_channels))

    sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=list(exclude_channels), preload=True)

    annot = mne.read_annotations(data[1])

    ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                 "Sleep stage 4": 4, "Sleep stage R": 5}
    #     # "Sleep stage ?": 5,
    #     # "Movement time": 5

    # 没有对DREAMS数据集进行数据切割
    '''
    annot.crop(annot[1]['onset'] - 30 * 60,
               annot[-2]['onset'] + 30 * 60)
    '''

    sleep_signals.set_annotations(annot, emit_warning=False)

    events, _ = mne.events_from_annotations(
        sleep_signals, event_id=ann2label, chunk_duration=30.)

    # Filtering
    tmax = 30. - 1. / sleep_signals.info['sfreq']

    if filter == True:
        sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

    # Breaking into Epochs
    epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                             event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                             on_missing='warn')
    # epochs = mne.make_fixed_length_epochs(sleep_signals, duration=duration, preload=True)

    sig_epochs = []
    label_epochs = []

    mean_epochs = []
    std_epochs = []

    signal_mean = np.mean(np.array([epochs_data]))
    signal_std = np.std(np.array([epochs_data]))

    for ep in range(len(epochs_data)):
        for sig in epochs_data[ep]:
            sig_epochs.append(sig)

        sleep_stage = epochs_data[ep].event_id

        if sleep_stage == {"Sleep stage W": 9}:
            label_epochs.append(0)
        if sleep_stage == {"Sleep stage 1": 1}:
            label_epochs.append(1)
        if sleep_stage == {"Sleep stage 2": 2}:
            label_epochs.append(2)
        if sleep_stage == {"Sleep stage 3": 3}:
            label_epochs.append(3)
        if sleep_stage == {"Sleep stage 4": 4}:
            label_epochs.append(3)
        if sleep_stage == {"Sleep stage R": 5}:
            label_epochs.append(4)

        mean_epochs.append(signal_mean)
        std_epochs.append(signal_std)

    sig_epochs = np.array(sig_epochs)
    mean_epochs = np.array(mean_epochs)
    std_epochs = np.array(std_epochs)
    label_epochs = np.array(label_epochs)

    main_ext_raw_data = sig_epochs
    main_labels = label_epochs
    main_sub_len = np.array([len(epochs_data)])
    main_mean = mean_epochs
    main_std = std_epochs

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std



# 针对figshare数据集，需要输入<PSG信号文件地址subjects>
# 返回PSG信号，长度，均值，标准差
def figshare_signal_extract(subjects, channel='eeg1', filter=True, freq=[0.2, 40]):
    all_channels = (
        'EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE', 'EEG F7-LE', 'EEG T3-LE',
        'EEG T5-LE', 'EEG Fz-LE', 'EEG Fp2-LE', 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 'EEG O2-LE',
        'EEG F8-LE', 'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE', 'EEG A2-A1', 'EEG 23A-23R',
        'EEG 24A-24R')

    data = [subjects]
    # eeg1代表'EEG Fz-LE'
    # eog代表'EEG Pz-LE'
    signal2idx = {"eeg1": 8, "eeg2": 1, "eog": 18, "emg": 3}

    all_channels_list = list(all_channels)
    all_channels_list.remove(all_channels[signal2idx[channel]])
    exclude_channels = tuple(all_channels_list)

    sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)

    # Filtering
    # tmax = 30. - 1. / sleep_signals.info['sfreq']

    if filter == True:
        sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

    duration = 30
    epochs = mne.make_fixed_length_epochs(sleep_signals, duration=duration, preload=True)

    # Calculate mean and std of the signal epochs
    signal_mean = np.mean(epochs)
    signal_std = np.std(epochs)

    main_ext_raw_data = epochs.get_data()
    main_sub_len = np.array([len(epochs)])
    main_mean = np.tile(signal_mean, (len(epochs), 1)).squeeze()
    main_std = np.tile(signal_std, (len(epochs), 1)).squeeze()

    return main_ext_raw_data, main_sub_len, main_mean, main_std


hospital_path = r"E:\\hkk\\项目_可解释睡眠分期\\项目数据集\\depression_2017-2019拼英\\"
def signal_extract_hospital(edf_anno_list, channel='eeg1', filter=True, freq=[0.2, 40], stride=3):

# 1.初始化无效数据、通道
    ignore_data = []

#【改通道】
    all_channels = (
        'F3', 'F4', 'C3', 'C4', 'O1', 'O2',
        'M1', 'M2',
        'E1', 'E2',
        'ECG2', 'ECG1',
        'Chin1', 'Chin2', 'LEG/L', 'LEG/R',
        'AIRFLOW', 'ABDO', 'THOR', 'Snore', 'Pos Sensor', 'Ox Status', 'Pulse', 'SpO2', 'Nasal Pressure', 'CPAP Flow',
        'CPAP Press', 'Pleth', 'Sum', 'PTT', 'Derived HR', 'Respiratory rate', 'Light', 'Manual Pos'
        )

    first_sub_flag = 0

    for edf in edf_anno_list:
            data = [ hospital_path + edf[0], hospital_path + edf[1]]
            print("preparing: " + data[0] + " " + data[1])

        # 【改数据获取】
            signal2idx = {"eeg1": 0, "eeg2": 1, "eeg3": 2, "eeg4": 3, "eeg5": 4, "eeg6": 5,
                          "eog1": 8, "eog2": 9}
            all_channels_list = list(all_channels)
            all_channels_list.remove(all_channels[signal2idx[channel]])
            exclude_channels = tuple(all_channels_list)

            sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)
            annot = mne.read_annotations(data[1])
            # print("Annotation descriptions:", annot.description)

        # 3.注释裁剪和事件生成
        # 【改映射】
        #     ann2label = {
        #         "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
        #          "Sleep stage R": 4, "Sleep stage ?": 5, "Movement time": 6}
            ann2label = {
                "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage 4": 4, "Sleep stage R": 5}

            ann2label_without_unknown_stages = {
                "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage 4": 4,
                "Sleep stage R": 5}

            # annot.crop(annot[1]['onset'] - 30 * 60, annot[-2]['onset'] + 30 * 60)

            sleep_signals.set_annotations(annot, emit_warning=False)

            events, _ = mne.events_from_annotations(
                sleep_signals, event_id=ann2label, chunk_duration=30.)


        # 4.信号过滤
            if filter == True:
                sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

        # 5.划分 Epoch
            tmax = 30. - 1. / sleep_signals.info['sfreq']
            epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                                     event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                                     on_missing='warn')

            epochs_data_without_unknown_stages = mne.Epochs(raw=sleep_signals, events=events,
                                                            event_id=ann2label_without_unknown_stages, tmin=0.,
                                                            tmax=tmax, baseline=None, preload=True, on_missing='warn')

            print(
                '===================================================================================================================================')
            print(
                f"                    Shape of Extracted Raw Signal for File {edf}                           ")
            print(
                f"                    Shape of Extracted Label for File {edf}                             ")
            # print('===================================================================================================================================')

            sig_epochs = []
            label_epochs = []

            mean_epochs = []
            std_epochs = []

            signal_mean = np.mean(np.array([epochs_data]))
            signal_std = np.std(np.array([epochs_data]))

            for ep in range(len(epochs_data)):
                for sig in epochs_data[ep]:
                    sig_epochs.append(sig)

                sleep_stage = epochs_data[ep].event_id

                if sleep_stage == {"Sleep stage W": 0}:
                    label_epochs.append(0)
                if sleep_stage == {"Sleep stage 1": 1}:
                    label_epochs.append(1)
                if sleep_stage == {"Sleep stage 2": 2}:
                    label_epochs.append(2)
                if sleep_stage == {"Sleep stage 3": 3}:
                    label_epochs.append(3)
                if sleep_stage == {"Sleep stage 4": 4}:
                    label_epochs.append(3)
                if sleep_stage == {"Sleep stage R": 5}:
                    label_epochs.append(4)

                mean_epochs.append(signal_mean)
                std_epochs.append(signal_std)

            sig_epochs = np.array(sig_epochs)
            mean_epochs = np.array(mean_epochs)
            std_epochs = np.array(std_epochs)
            label_epochs = np.array(label_epochs)

            if first_sub_flag == 0:
                main_ext_raw_data = sig_epochs
                main_labels = label_epochs
                main_sub_len = np.array([len(epochs_data)])
                main_mean = mean_epochs
                main_std = std_epochs
                first_sub_flag = 1
            else:
                main_ext_raw_data = np.concatenate((main_ext_raw_data, sig_epochs), axis=0)
                main_labels = np.concatenate((main_labels, label_epochs), axis=0)
                main_sub_len = np.concatenate((main_sub_len, np.array([len(epochs_data)])), axis=0)
                main_mean = np.concatenate((main_mean, mean_epochs), axis=0)
                main_std = np.concatenate((main_std, std_epochs), axis=0)

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std




# 主函数，封装数据为h5py文件保存
def main():
    args = parse_option()

    # Separate Subjects into 5 groups
    from sklearn.model_selection import KFold
    days = np.arange(1, 3)
    subjects = np.arange(0, 83)
    print(f"Subjects : {subjects}")
    print(f"Days : {days}")

    fivefold_list = []
    kf = KFold(n_splits=5, shuffle=True  # 5, 2
               , random_state=2
               )

    sub_1, sub_2, sub_3, sub_4, sub_5 = kf.split(subjects)
    sub_1 = sub_1[1]
    sub_2 = sub_2[1]
    sub_3 = sub_3[1]
    sub_4 = sub_4[1]
    sub_5 = sub_5[1]

    print(f"Subjects Group 1 : {sub_1}")
    print(f"Subjects Group 2 : {sub_2}")
    print(f"Subjects Group 3 : {sub_3}")
    print(f"Subjects Group 4 : {sub_4}")
    print(f"Subjects Group 5 : {sub_5}")

    for i in sub_1:
        if i in subjects:
            subjects[i] = 0
        else:
            print("Error")

    for i in sub_2:
        if i in subjects:
            subjects[i] = 0
        else:
            print("Error")

    for i in sub_3:
        if i in subjects:
            subjects[i] = 0
        else:
            print("Error")

    for i in sub_4:
        if i in subjects:
            subjects[i] = 0
        else:
            print("Error")

    for i in sub_5:
        if i in subjects:
            subjects[i] = 0
        else:
            print("Error")

    print(subjects)

    # ==============================================================>
    # Change Channels to extract data for other PSG channels 'eeg1', ''eog', 'eeg2'
    # ==============================================================>

    # ==============================================================>
    # For 'eeg1'
    # ==============================================================>

    ## Save Path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    eeg1_1, labels_1, len_1, eeg1_m1, eeg1_std1 = signal_extract(sub_1, days, channel='eeg1', filter=True,
                                                                 freq=[0.2, 40])
    print(f"Train data shape : {eeg1_1.shape}, Train label shape : {labels_1.shape}")

    #### Save data as .h5. ######
    hf = h5py.File(f'{save_path}/x1.h5', 'w')
    hf.create_dataset('data', data=eeg1_1)
    hf.close()
    hf = h5py.File(f'{save_path}/y1.h5', 'w')
    hf.create_dataset('data', data=labels_1)
    hf.close()
    hf = h5py.File(f'{save_path}/mean1.h5', 'w')
    hf.create_dataset('data', data=eeg1_m1)
    hf.close()
    hf = h5py.File(f'{save_path}/std1.h5', 'w')
    hf.create_dataset('data', data=eeg1_std1)
    hf.close()

    eeg1_2, labels_2, len_2, eeg1_m2, eeg1_std2 = signal_extract(sub_2, days, channel='eeg1', filter=True,
                                                                 freq=[0.2, 40])
    print(f"Train data shape : {eeg1_2.shape}, Train label shape : {labels_2.shape}")

    #### Save data as .h5. ######
    hf = h5py.File(f'{save_path}/x2.h5', 'w')
    hf.create_dataset('data', data=eeg1_2)
    hf.close()
    hf = h5py.File(f'{save_path}/y2.h5', 'w')
    hf.create_dataset('data', data=labels_2)
    hf.close()
    hf = h5py.File(f'{save_path}/mean2.h5', 'w')
    hf.create_dataset('data', data=eeg1_m2)
    hf.close()
    hf = h5py.File(f'{save_path}/std2.h5', 'w')
    hf.create_dataset('data', data=eeg1_std2)
    hf.close()

    eeg1_3, labels_3, len_3, eeg1_m3, eeg1_std3 = signal_extract(sub_3, days, channel='eeg1', filter=True,
                                                                 freq=[0.2, 40])
    print(f"Train data shape : {eeg1_3.shape}, Train label shape : {labels_3.shape}")

    #### Save data as .h5. ######
    hf = h5py.File(f'{save_path}/x3.h5', 'w')
    hf.create_dataset('data', data=eeg1_3)
    hf.close()
    hf = h5py.File(f'{save_path}/y3.h5', 'w')
    hf.create_dataset('data', data=labels_3)
    hf.close()
    hf = h5py.File(f'{save_path}/mean3.h5', 'w')
    hf.create_dataset('data', data=eeg1_m3)
    hf.close()
    hf = h5py.File(f'{save_path}/std3.h5', 'w')
    hf.create_dataset('data', data=eeg1_std3)
    hf.close()

    eeg1_4, labels_4, len_4, eeg1_m4, eeg1_std4 = signal_extract(sub_4, days, channel='eeg1', filter=True,
                                                                 freq=[0.2, 40])
    print(f"Train data shape : {eeg1_4.shape}, Train label shape : {labels_4.shape}")

    #### Save data as .h5. ######
    hf = h5py.File(f'{save_path}/x4.h5', 'w')
    hf.create_dataset('data', data=eeg1_4)
    hf.close()
    hf = h5py.File(f'{save_path}/y4.h5', 'w')
    hf.create_dataset('data', data=labels_4)
    hf.close()
    hf = h5py.File(f'{save_path}/mean4.h5', 'w')
    hf.create_dataset('data', data=eeg1_m4)
    hf.close()
    hf = h5py.File(f'{save_path}/std4.h5', 'w')
    hf.create_dataset('data', data=eeg1_std4)
    hf.close()

    eeg1_5, labels_5, len_5, eeg1_m5, eeg1_std5 = signal_extract(sub_5, days, channel='eeg1', filter=True,
                                                                 freq=[0.2, 40])
    print(f"Train data shape : {eeg1_5.shape}, Train label shape : {labels_5.shape}")

    #### Save data as .h5. ######
    hf = h5py.File(f'{save_path}/x5.h5', 'w')
    hf.create_dataset('data', data=eeg1_5)
    hf.close()
    hf = h5py.File(f'{save_path}/y5.h5', 'w')
    hf.create_dataset('data', data=labels_5)
    hf.close()
    hf = h5py.File(f'{save_path}/mean5.h5', 'w')
    hf.create_dataset('data', data=eeg1_m5)
    hf.close()
    hf = h5py.File(f'{save_path}/std5.h5', 'w')
    hf.create_dataset('data', data=eeg1_std5)
    hf.close()

    # ==============================================================>
    # For 'eog'
    # ==============================================================>

    eog1, labels_1, len_1, eog_m1, eog_std1 = signal_extract(sub_1, days, channel='eog', filter=True, freq=[0.2, 40])
    print(f"Train data shape : {eog1.shape}, Train label shape : {labels_1.shape}")

    #### Save data as .h5. ######
    #### Save data as .h5. ######
    hf = h5py.File(f'{save_path}/eog1.h5', 'w')
    hf.create_dataset('data', data=eog1)
    hf.close()
    hf = h5py.File(f'{save_path}/eog_m1.h5', 'w')
    hf.create_dataset('data', data=eog_m1)
    hf.close()
    hf = h5py.File(f'{save_path}/eog_std1.h5', 'w')
    hf.create_dataset('data', data=eog_std1)
    hf.close()

    eog2, labels_2, len_2, eog_m2, eog_std2 = signal_extract(sub_2, days, channel='eog', filter=True, freq=[0.2, 40])
    print(f"Train data shape : {eog2.shape}, Train label shape : {labels_2.shape}")

    #### Save data as .h5. ######
    #### Save data as .h5. ######
    hf = h5py.File(f'{save_path}/eog2.h5', 'w')
    hf.create_dataset('data', data=eog2)
    hf.close()
    hf = h5py.File(f'{save_path}/eog_m2.h5', 'w')
    hf.create_dataset('data', data=eog_m2)
    hf.close()
    hf = h5py.File(f'{save_path}/eog_std2.h5', 'w')
    hf.create_dataset('data', data=eog_std2)
    hf.close()

    eog3, labels_3, len_3, eog_m3, eog_std3 = signal_extract(sub_3, days, channel='eog', filter=True, freq=[0.2, 40])
    print(f"Train data shape : {eog3.shape}, Train label shape : {labels_3.shape}")

    #### Save data as .h5. ######
    hf = h5py.File(f'{save_path}/eog3.h5', 'w')
    hf.create_dataset('data', data=eog3)
    hf.close()
    hf = h5py.File(f'{save_path}/eog_m3.h5', 'w')
    hf.create_dataset('data', data=eog_m3)
    hf.close()
    hf = h5py.File(f'{save_path}/eog_std3.h5', 'w')
    hf.create_dataset('data', data=eog_std3)
    hf.close()

    eog4, labels_4, len_4, eog_m4, eog_std4 = signal_extract(sub_4, days, channel='eog', filter=True, freq=[0.2, 40])
    print(f"Train data shape : {eog4.shape}, Train label shape : {labels_4.shape}")

    #### Save data as .h5. ######
    hf = h5py.File(f'{save_path}/eog4.h5', 'w')
    hf.create_dataset('data', data=eog4)
    hf.close()
    hf = h5py.File(f'{save_path}/eog_m4.h5', 'w')
    hf.create_dataset('data', data=eog_m4)
    hf.close()
    hf = h5py.File(f'{save_path}/eog_std4.h5', 'w')
    hf.create_dataset('data', data=eog_std4)
    hf.close()

    eog5, labels_5, len_5, eog_m5, eog_std5 = signal_extract(sub_5, days, channel='eog', filter=True, freq=[0.2, 40])
    print(f"Train data shape : {eeg1_5.shape}, Train label shape : {labels_5.shape}")

    #### Save data as .h5. ######
    hf = h5py.File(f'{save_path}/eog5.h5', 'w')
    hf.create_dataset('data', data=eog5)
    hf.close()
    hf = h5py.File(f'{save_path}/eog_m5.h5', 'w')
    hf.create_dataset('data', data=eog_m5)
    hf.close()
    hf = h5py.File(f'{save_path}/eog_std5.h5', 'w')
    hf.create_dataset('data', data=eog_std5)
    hf.close()



if __name__ == '__main__':
    main()