# !pip install mne
import fnmatch
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import mne
from sklearn.model_selection import KFold
# from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import h5py

#path必须以\\结尾，否则传入的路径与文件名拼接错误
hospital_path = "C:\\Users\\LYT\\Desktop\\数据集\\hospital_depression\\depression_pinyin - 副本\\"

def signal_extract_sequential_hospital(edf_anno_list, channel='eeg1', filter=True, freq=[0.2, 40], stride=3):

# 1.初始化无效数据、通道
    ignore_data = []

#【改通道】
    all_channels = (
        'F3', 'F4', 'C3', 'C4', 'O1', 'O2',
        'M1', 'M2',
        'E1', 'E2',
        'ECG1', 'ECG2',
        'Chin1', 'Chin2', 'Chin3', 'LEG/L', 'LEG/R',
        'Airflow', 'Abdo', 'Thor', 'Snore', 'Sum', 'PosSensor', 'Ox Status', 'Pulse', 'SpO2', 'Nasal Pressure', 'CPAP Flow',
        'CPAP Press', 'Pleth', 'Sum', 'Derived HR', 'Light', 'Manual Pos', 'Respiratory Rate'
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

            for ep in range(len(epochs_data) - (stride - 1)):

                existing_labels = epochs_data[ep:ep + stride].event_id

                if (not ('Sleep stage ?' in existing_labels or 'Movement time' in existing_labels)):

                    temp_epochs = []
                    temp_labels = []

                    for i in range(ep, ep + stride):

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
                                temp_labels.append(5)

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


def main():
    # Separate Subjects into 5 groups
    # 【改】======================================================================

    edf_anno_list = []
    # 查找所有的edf文件和对应的注释文件【列表edf_anno_list】
    for filename in os.listdir(hospital_path):
        if fnmatch.fnmatch(filename, '*_mne_Annotation.txt'):  # 查找以 "_mne_Annotation.txt" 结尾的文件
            tmp = filename.split('_mne_Annotation.txt')[0]  # 提取文件名前缀部分，比如 chenfang20170510
            for i in os.listdir(hospital_path):
                if fnmatch.fnmatch(i, tmp + '.edf'):  # 查找以相同前缀并以 .edf 结尾的文件，比如chenfang20170510.edf
                    edf_anno_list.append((i, filename))  # 将edf文件和注释文件的元组添加到列表中
                    print(i + " matched " + filename)

    # 5折交叉验证
    fivefold_list = []
    kf = KFold(n_splits=5, shuffle=True, random_state=2)

    ## Save Path
    save_path = f"./extract_dataset_multi_epoch_hospital"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 使用 KFold 来划分数据
    '''
        for train_index, test_index in kf.split(edf_anno_list):
            fivefold_list.append((train_index, test_index))
        print(fivefold_list)
    '''

    sub1, sub2, sub3, sub4, sub5 = kf.split(edf_anno_list)

    sub1 = sub1[1]
    sub_1 = [edf_anno_list[i] for i in sub1]

    sub2 = sub2[1]
    sub_2 = [edf_anno_list[i] for i in sub2]

    sub3 = sub3[1]
    sub_3 = [edf_anno_list[i] for i in sub3]

    sub4 = sub4[1]
    sub_4 = [edf_anno_list[i] for i in sub4]

    sub5 = sub5[1]
    sub_5 = [edf_anno_list[i] for i in sub5]

    print(f"Subjects Group 1 : {sub_1}")
    print(f"Subjects Group 2 : {sub_2}")
    print(f"Subjects Group 3 : {sub_3}")
    print(f"Subjects Group 4 : {sub_4}")
    print(f"Subjects Group 5 : {sub_5}")

    # save #
    eeg1_1, labels_1, len_1, eeg1_m1, eeg1_std1 = signal_extract_sequential_hospital(sub_1, channel='eeg1', filter=True,
                                                                                     freq=[0.2, 40], stride=5)
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

    eeg1_2, labels_2, len_2, eeg1_m2, eeg1_std2 = signal_extract_sequential_hospital(sub_2, channel='eeg1', filter=True,
                                                                                     freq=[0.2, 40], stride=5)
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

    eeg1_3, labels_3, len_3, eeg1_m3, eeg1_std3 = signal_extract_sequential_hospital(sub_3, channel='eeg1', filter=True,
                                                                                     freq=[0.2, 40], stride=5)
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

    eeg1_4, labels_4, len_4, eeg1_m4, eeg1_std4 = signal_extract_sequential_hospital(sub_4, channel='eeg1', filter=True,
                                                                                     freq=[0.2, 40], stride=5)
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

    eeg1_5, labels_5, len_5, eeg1_m5, eeg1_std5 = signal_extract_sequential_hospital(sub_5, channel='eeg1', filter=True,
                                                                                     freq=[0.2, 40], stride=5)
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


    eog1, labels_1, len_1, eog_m1, eog_std1 = signal_extract_sequential_hospital(sub_1, channel='eog', filter=True,
                                                                        freq=[0.2, 40], stride=5)
    print(f"Train data shape : {eog1.shape}, Train label shape : {labels_1.shape}")

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

    eog2, labels_2, len_2, eog_m2, eog_std2 = signal_extract_sequential_hospital(sub_2, channel='eog', filter=True,
                                                                        freq=[0.2, 40], stride=5)
    print(f"Train data shape : {eog2.shape}, Train label shape : {labels_2.shape}")

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

    eog3, labels_3, len_3, eog_m3, eog_std3 = signal_extract_sequential_hospital(sub_3, channel='eog', filter=True,
                                                                        freq=[0.2, 40], stride=5)
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

    eog4, labels_4, len_4, eog_m4, eog_std4 = signal_extract_sequential_hospital(sub_4, channel='eog', filter=True,
                                                                        freq=[0.2, 40], stride=5)
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

    eog5, labels_5, len_5, eog_m5, eog_std5 = signal_extract_sequential_hospital(sub_5, channel='eog', filter=True,
                                                                        freq=[0.2, 40], stride=5)
    print(f"Train data shape : {eog5.shape}, Train label shape : {labels_5.shape}")

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

