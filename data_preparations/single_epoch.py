# !pip install mne
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import mne
from mne.datasets.sleep_physionet.age import fetch_data
# from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import h5py
import argparse


# 对于一个数据生成程序，可以通过命令行指定存储数据集的路径【不需要修改程序代码】
def parse_option():
    # 创建命令行参数解析器对象 parser，并提供一个描述【在使用 --help 选项时显示】
    parser = argparse.ArgumentParser('Argument for data generation')

    # 添加一个名为 --save_path 的命令行参数
    # 设定参数【数据类型为字符串、默认值、help信息】
    parser.add_argument('--save_path', type=str, default='./extract_dataset_single_epoch',
                        help='Path to store project results')

    # 解析器 parser 对命令行参数进行解析，将解析结果存储在 opt 对象中
    opt = parser.parse_args()

    return opt


def signal_extract(subjects, days, channel='eeg1', filter=True, freq=[0.2, 40]):

# 1.初始化无效数据、通道
    ignore_data = [[13, 2], [36, 1], [39, 1], [39, 2], [52, 1], [68, 1], [68, 2], [69, 1], [69, 2], [78, 1], [78, 2],[79, 1], [79, 2]]
    all_channels = ('EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    # 一个标志，用于判断是否是第一个处理的受试者
    first_sub_flag = 0

    for sub in subjects:
        for day_ in days:
            if [sub, day_] in ignore_data:
                continue
# 2.数据获取和通道处理
            # fetch_data 函数，参数输入2个只含1个元素的列表，返回值是1个只含1个元素的列表[data]
            # data是一个列表，包含2个元素，都是一个EDF文件的路径
                # data[0] 指向【PSG数据文件】
                # data[1] 指向【睡眠阶段标注文件】
            # 根据受试者编号、记录天数编号，从sleepEDF下载数据
            [data] = fetch_data(subjects=[sub], recording=[day_])

            # 将不同的通道名称映射到相应的索引（0、1、2、3）【以便在 all_channels 列表中找到对应的通道位置】
            signal2idx = {"eeg1": 0, "eeg2": 1, "eog": 2, "emg": 3}

            # 得到排除的通道的列表exclude_channels
            all_channels_list = list(all_channels)
            all_channels_list.remove( all_channels[ signal2idx[channel] ] )
            exclude_channels = tuple(all_channels_list)

            # 从指定受试者指定天数的PSG文件路径中，读取非排除的1个通道的信号到sleep_signals
                # verbose=True：在读取文件时输出详细的信息
                # preload=True：将文件数据预加载到内存中
            sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)

# 3.注释裁剪和事件生成
            # 从指定受试者指定天数的标注文件路径中，读取睡眠分期标注到annot
            annot = mne.read_annotations(data[1])

            # 将不同的分期标签映射到相应的索引（0、1、2、3、4、5）【以便在】
                # Movement time：运动时间，指在睡眠期间的任何体动
            ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                         "Sleep stage 4": 4, "Sleep stage R": 5}
            #     # "Sleep stage ?": 5,
            #     # "Movement time": 5

            # crop函数，参数是裁剪后的开始时间、结束时间
                # 减去30分钟后开始时间会变为负数, 但后面使用的库函数会把最小的数视为0来处理
            # 裁剪是为了确保标注的完整性
            annot.crop(annot[1]['onset'] - 30 * 60,
                              annot[-2]['onset'] + 30 * 60)

            # 将裁剪后的标注信息加入到信号对象sleep_signals中
                # emit_warning=False：表示在设置注释时不发出警告信息
            sleep_signals.set_annotations(annot, emit_warning=False)

            # events_from_annotations函数，参数：信号对象、事件id与原始标注的映射关系、事件元素的时长
            # 根据信号对象、注释信息生成事件数组events
            events, _ = mne.events_from_annotations(
                sleep_signals, event_id=ann2label, chunk_duration=30.)

# 4.信号过滤

            # 如果设置为需要滤波
                # filter函数，对数据信号的副本滤波，参数：设定的上下截止频率
            if filter == True:
                sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

# 5.划分 Epoch 并提取数据
#把信号对象sleep_signals，根据事件数组events，划分成epoch对象数组

            # Epoches函数，将原始信号对象切割为epoch对象数组，参数：原始信号、事件数组、事件id与标注的映射、
                # 每个epoch相对事件开始时间的时间起点【tmin = 0】、
                # 每个epoch相对事件开始时间的时间终点【每个epoch的时长为 30 秒减去一个采样点，避免边界问题】、
                # 是否对信号进行基线校正
                # 是否预加载到内存
                # 遇到缺失的事件或其他问题，会发出警告
            tmax = 30. - 1. / sleep_signals.info['sfreq']
            epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                                     event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                                     on_missing='warn')

            # epoch信号对象列表、事件id列表
            sig_epochs = []
            label_epochs = []

            # 每个epoch的信号均值和标准差
            mean_epochs = []
            std_epochs = []

            signal_mean = np.mean(np.array([epochs_data]))
            signal_std = np.std(np.array([epochs_data]))

# 6.标签分配和数据存储
# 一次处理全部epoch【得到sig_epochs列表，label_epochs列表，形状都为 (1, n_epoches)】
            for ep in range(len(epochs_data)):
                # 将epoch中的每个采样点信号数据提取为sig数组【形状为 (1, n_samples)】
                    # sig_epochs 列表，元素为每个epoch的信号数据数组【形状为 (1, n_epoches)】
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

# 7.拼接数据
            # 列表都转化成数组
            sig_epochs = np.array(sig_epochs)
            mean_epochs = np.array(mean_epochs)
            std_epochs = np.array(std_epochs)
            label_epochs = np.array(label_epochs)

            # 如果处理的是第一个受试者，要初始化4个存储变量
            if first_sub_flag == 0:
                main_ext_raw_data = sig_epochs    # 【所有受试者的切割成epoches的信号数据】
                main_labels = label_epochs    # 所有受试者的标签数据
                main_sub_len = np.array([len(epochs_data)])    # 每个受试者的 epoch 数量
                main_mean = mean_epochs    # 所有受试者的信号均值数据
                main_std = std_epochs      # 所有受试者的信号标准差数据
                first_sub_flag = 1
            else:
                main_ext_raw_data = np.concatenate((main_ext_raw_data, sig_epochs), axis=0)
                main_labels = np.concatenate((main_labels, label_epochs), axis=0)
                main_sub_len = np.concatenate((main_sub_len, np.array([len(epochs_data)])), axis=0)
                main_mean = np.concatenate((main_mean, mean_epochs), axis=0)
                main_std = np.concatenate((main_std, std_epochs), axis=0)

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std


def main():
# 1.从控制台获取命令参数【生成的数据集的保存路径】
    args = parse_option()

# 2.使用KFold，将受试者随机分为5组【5折交叉验证】【取1个作为验证集，4个作为训练集】
    from sklearn.model_selection import KFold
    days = np.arange(1, 3)
    subjects = np.arange(0, 83)    # 0-82号受试者的标记位【如果该受试者被已被用于交叉验证，就标记为1】
    print(f"Subjects : {subjects}")
    print(f"Days : {days}")

    fivefold_list = []     # 用来存储每一折数据的数组索引
    # 创建一个 KFold 对象，参数：划分成几折，划分前是否打乱、设置随机数种子【确保每次划分相同】
    kf = KFold(n_splits=5, shuffle=True  # 5, 2
               , random_state=2
               )

    #将受试者分成5组，并存储每一组的受试者编号
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

# 3.将已经被用作交叉验证的受试者，subjects标记位标记为1
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

# 4.提取并切割，指定通道、受试者组的信号【eeg1、eog】、【sub_1、sub_2、sub_3、sub_4、sub_5】
    # 对第一组受试者的原始数据，做预处理，【返回：epoch数组、标签数组、epoch数量、信号均值、信号标准差】
    eeg1_1, labels_1, len_1, eeg1_m1, eeg1_std1 = signal_extract(sub_1, days, channel='eeg1', filter=True,
                                                                 freq=[0.2, 40])
    print(f"Train data shape : {eeg1_1.shape}, Train label shape : {labels_1.shape}")


# 5.将得到的数据集的各种信息的数组，保存进 HDF5 文件中
    hf = h5py.File(f'{save_path}/x1.h5', 'w')      # epoch数组【x1】
    hf.create_dataset('data', data=eeg1_1)
    hf.close()
    hf = h5py.File(f'{save_path}/y1.h5', 'w')      # 标签数组【y1】
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
