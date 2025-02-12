import os
import numpy as np
import matplotlib.pyplot as plt
import random
import mne
from mne.datasets.sleep_physionet.age import fetch_data
# from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import h5py


# 不同【数据提取函数，新增参数：stride】
def signal_extract_sequential(subjects, days, channel='eeg1', filter=True, freq=[0.2, 40], stride=3):
    ignore_data = [[13, 2], [36, 1], [39, 1], [39, 2], [52, 1], [68, 1], [68, 2], [69, 1], [69, 2], [78, 1], [78, 2],
                   [79, 1], [79, 2]]
    all_channels = (
        'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    first_sub_flag = 0
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

            ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                         "Sleep stage 4": 4,
                         "Sleep stage R": 5, "Sleep stage ?": 6, "Movement time": 7}

# 不同【剔除了 未知、无效睡眠阶段的 映射】======================================================================================
            ann2label_without_unknown_stages = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
                                                "Sleep stage 3": 3, "Sleep stage 4": 4,
                                                "Sleep stage R": 5}

            annot.crop(annot[1]['onset'] - 30 * 60,
                       annot[-2]['onset'] + 30 * 60)

            sleep_signals.set_annotations(annot, emit_warning=False)

            events, _ = mne.events_from_annotations(
                sleep_signals, event_id=ann2label, chunk_duration=30.)

            #Filtering
            tmax = 30. - 1. / sleep_signals.info['sfreq']

            if filter == True:
                sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

            epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                                     event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                                     on_missing='warn')

 # 不同【提取数据，不提取注释为"Sleep stage ?" 或 "Movement time"）的epoch】【得到干净的epoch集合】================================================
            epochs_data_without_unknown_stages = mne.Epochs(raw=sleep_signals, events=events,
                                                            event_id=ann2label_without_unknown_stages, tmin=0.,
                                                            tmax=tmax, baseline=None, preload=True, on_missing='warn')

            print(
                '===================================================================================================================================')
            print(
                f"                    Shape of Extracted Raw Signal for Subject {sub} and Day {day_}                          ")
            print(
                f"                    Shape of Extracted Label for Subject {sub} and Day {day_}                             ")
            print(
                '===================================================================================================================================')

            sig_epochs = []
            label_epochs = []

            mean_epochs = []
            std_epochs = []

            signal_mean = np.mean(np.array([epochs_data_without_unknown_stages]))
            signal_std = np.std(np.array([epochs_data_without_unknown_stages]))

# 6.标签分配和数据存储=============================================================================
# 不同【一次处理一个序列内的epoch】
#【得到的sig_epochs矩阵，每行为stride个epoch的 sig 数组（一个序列的信号），形状为 (n_sequences, stride)】
#【得到的label_epochs矩阵，每行为stride个epoch的 label 数组（一个序列的标签），形状 (n_sequences, stride)】
            # 每次提取一个序列【ep：当前序列的起始epoch的索引】
            for ep in range(len(epochs_data) - (stride - 1)):

                # existing_labels：当前序列的epoch标签数组
                existing_labels = epochs_data[ep:ep + stride].event_id

                # 如果序列的标签数组包含无效标签，就跳过下面的处理【跳过这个序列】
                if (not ('Sleep stage ?' in existing_labels or 'Movement time' in existing_labels)):
                    temp_epochs = []  # 当前序列的epoch数组
                    temp_labels = []  # 当前序列的标签数组

                    # 遍历当前序列中的epoch【i：每个epoch的索引】
                    for i in range(ep, ep + stride):

                        # 当前epoch中的每个采样点信号数据=sig数组【形状为 (1, n_samples)】【n_channels=1】
                            # temp_epochs 列表，元素为每个epoch的sig数组【形状为 (1, n_epoches)】
                        for sig in epochs_data[i]:
                            temp_epochs.append(sig[0])

                            sleep_stage = epochs_data[i].event_id
                            if sleep_stage == {"Sleep stage W": 0}:
                                temp_labels.append(0)
                            if sleep_stage == {"Sleep stage 1": 1}:
                                temp_labels.append(1)
                            if sleep_stage == {"Sleep stage 2": 2}:
                                temp_labels.append(2)
                            if sleep_stage == {"Sleep stage 3": 3}:
                                temp_labels.append(3)
                            if sleep_stage == {"Sleep stage 4": 4}:
                                temp_labels.append(3)
                            if sleep_stage == {"Sleep stage R": 5}:
                                temp_labels.append(4)

                    sig_epochs.append(temp_epochs)
                    label_epochs.append(temp_labels)
                    mean_epochs.append(signal_mean)
                    std_epochs.append(signal_std)

            sig_epochs = np.array(sig_epochs)
            mean_epochs = np.array(mean_epochs)
            std_epochs = np.array(std_epochs)
            label_epochs = np.array(label_epochs)

            if first_sub_flag == 0:
                main_ext_raw_data = sig_epochs
                main_labels = label_epochs
                main_sub_len = np.array([len(label_epochs)])
                main_mean = mean_epochs
                main_std = std_epochs
                first_sub_flag = 1
            else:
                main_ext_raw_data = np.concatenate((main_ext_raw_data, sig_epochs), axis=0)
                main_labels = np.concatenate((main_labels, label_epochs), axis=0)
                main_sub_len = np.concatenate((main_sub_len, np.array([len(label_epochs)])), axis=0)
                main_mean = np.concatenate((main_mean, mean_epochs), axis=0)
                main_std = np.concatenate((main_std, std_epochs), axis=0)

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std


import os
import numpy as np
import matplotlib.pyplot as plt
import random
import mne
from mne.datasets.sleep_physionet.age import fetch_data
# from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import h5py


def signal_extract(subjects, days, channel='eeg1', filter=True, freq=[0.2, 40]):
    ignore_data = [[13, 2], [36, 1], [39, 1], [39, 2], [52, 1], [68, 1], [68, 2], [69, 1], [69, 2], [78, 1], [78, 2],
                   [79, 1], [79, 2]]
    all_channels = (
        'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    first_sub_flag = 0
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

            ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                         "Sleep stage 4": 4, "Sleep stage R": 5}
            #     # "Sleep stage ?": 5,
            #     # "Movement time": 5

            annot.crop(annot[1]['onset'] - 30 * 60,
                       annot[-2]['onset'] + 30 * 60)

            sleep_signals.set_annotations(annot, emit_warning=False)

            events, _ = mne.events_from_annotations(
                sleep_signals, event_id=ann2label, chunk_duration=30.)

            #Filtering
            tmax = 30. - 1. / sleep_signals.info['sfreq']

            if filter == True:
                sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

            #Breaking into Epochs
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
