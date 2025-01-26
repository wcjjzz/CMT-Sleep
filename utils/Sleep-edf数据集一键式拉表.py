## 批量测试脚本
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import mne
from mne.datasets.sleep_physionet.age import fetch_data

import torch
from torchvision import transforms, datasets
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from datetime import date
from utils.metrics import accuracy, kappa, g_mean, plot_confusion_matrix, confusion_matrix
import pandas as pd
from models.sequence_cmt import Seq_Cross_Transformer_Network #as Seq_Cross_Transformer_Network
from models.sequence_cmt import Epoch_Cross_Transformer
from models.model_blocks import PositionalEncoding, Window_Embedding, Intra_modal_atten, Cross_modal_atten, Feed_forward

import warnings

warnings.filterwarnings("ignore")




def new_signal_extract(edf_path, hypnogram_path, channel='eeg1', filter=True, freq=[0.2, 40]):
    all_channels = (
        'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    first_sub_flag = 0

    data = [edf_path, hypnogram_path]
    print("preparing: " + data[0] + " " + data[1])
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

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std


def signal_extract(subjects, days, channel='eeg1', filter=True, freq=[0.2, 40]):
    ignore_data = [[13, 2], [36, 1], [39, 1], [39, 2], [52, 1], [68, 1], [68, 2], [69, 1], [69, 2], [78, 1], [78, 2], [79, 1], [79, 2]]
    all_channels = (
        'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    first_sub_flag = 0
    for sub in subjects:
        for day_ in days:
            if [sub, day_] in ignore_data:
                continue
            [data] = fetch_data(subjects=[sub], recording=[day_])
            print("preparing: " + data[0] + " " + data[1])
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


def statistic_append(result_path, subject):
    data = {
        "测评对象": [subject],
        "准确率": [acc],
        "Kappa系数": [kap],
        "宏平均F1分数": [f1],
        "G均值": [g],
        "敏感性\真正率": [sens],
        "特异性": [spec],
        "类别F1分数1": f1_l[0],
        "类别F1分数2": f1_l[1],
        "类别F1分数3": f1_l[2],
        "类别F1分数4": f1_l[3],
        "类别F1分数5": f1_l[4],
    }
    df_new = pd.DataFrame(data)

    # 检查文件是否存在
    if not os.path.isfile(result_path):
        # 文件不存在，直接写入新数据
        df_new.to_excel(result_path, index=False)
        print("A new file has been created.")
    else:
        # 文件存在，追加新数据
        try:
            df_existing = pd.read_excel(result_path)
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            df_updated.to_excel(result_path, index=False)
            print("Data has been appended to file.")
        except Exception as e:
            # 如果出现其他异常，打印错误信息并退出
            print(f"An error occurred: {e}")
            raise
class SleepEDF_Seq_MultiChan_Dataset_Inference(Dataset):
    def __init__(self, eeg_file, eog_file, label_file, device, mean_eeg_l=None, sd_eeg_l=None,
                 mean_eog_l=None, sd_eog_l=None, mean_eeg2_l=None, sd_eeg2_l=None, transform=None,
                 target_transform=None, sub_wise_norm=False, num_seq=5):
        """

        """
        # Get the data

        self.eeg = eeg_file
        self.eog = eog_file
        self.labels = label_file

        self.labels = torch.from_numpy(self.labels)

        bin_labels = np.bincount(self.labels)
        print(f"Labels count: {bin_labels}")
        print(f"Shape of EEG : {self.eeg.shape} , EOG : {self.eog.shape}")  # , EMG: {self.eeg2.shape}")
        print(f"Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
            print(f"Reading Subject wise mean and sd")

            self.mean_eeg = mean_eeg_l
            self.sd_eeg = sd_eeg_l
            self.mean_eog = mean_eog_l
            self.sd_eog = sd_eog_l

        self.sub_wise_norm = sub_wise_norm
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.num_seq = num_seq

    def __len__(self):
        return len(self.labels) - self.num_seq

    def __getitem__(self, idx):
        eeg_data = self.eeg[idx:idx + self.num_seq].squeeze()
        eog_data = self.eog[idx:idx + self.num_seq].squeeze()
        label = self.labels[idx:idx + self.num_seq, ]

        if self.sub_wise_norm == True:
            eeg_data = (eeg_data - self.mean_eeg[idx]) / self.sd_eeg[idx]
            eog_data = (eog_data - self.mean_eog[idx]) / self.sd_eog[idx]
        elif self.mean and self.sd:
            eeg_data = (eeg_data - self.mean[0]) / self.sd[0]
            eog_data = (eog_data - self.mean[1]) / self.sd[1]
        if self.transform:
            eeg_data = self.transform(eeg_data)
            eog_data = self.transform(eog_data)
        if self.target_transform:
            label = self.target_transform(label)
        return eeg_data, eog_data, label


model_path = './checkpoint_model_best_acc.pth.tar'
#以下两行仅在use_online_mne = False时起作用
# edf_path = r'C:\Users\Tian_Yumi\mne_data\physionet-sleep-data\SC4381F0-PSG.edf'
# hypnogram_path = r'C:\Users\Tian_Yumi\mne_data\physionet-sleep-data\SC4381FC-Hypnogram.edf'
result_path = f"results/statistic.xlsx"


# result_path = r"D:\AIGC\CMT-Sleep\results"
# model_path = r"D:\AIGC\CMT-Sleep\models\1028_8ps_seq\checkpoint_model_best_acc.pth.tar"

use_online_mne = True
subjects = [x for x in range(13, 83)]  # 0-82 (inclusive)
days = [1, 2]
ignore_data = [[13, 2], [36, 1], [39, 1], [39, 2], [52, 1], [68, 1], [68, 2], [69, 1], [69, 2], [78, 1], [78, 2], [79, 1], [79, 2]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("decive: ", device)
test_model = torch.load(model_path, map_location=device)
test_model.eval()  # 设置为评估模式
for sub in subjects:
    for day_ in days:
        if [sub,day_]  in ignore_data:
            continue
        eeg1_1, labels_1, len_1, eeg1_m1, eeg1_std1 = signal_extract([sub], [day_], channel='eeg1', filter=True,
                                                                     freq=[0.2, 40])
        eog_1, _, len_1, eog_m1, eog_std1 = signal_extract([sub], [day_], channel='eog', filter=True, freq=[0.2, 40])
        num_seq = 15
        print("PRINT", eeg1_1.shape, labels_1.shape, len_1.shape, eeg1_m1.shape, eeg1_std1.shape)
        print("PRINT", eog_1.shape, len_1.shape, eog_m1.shape, eog_std1.shape)
        print("Data Preparation Done!")

        infer_dataset = SleepEDF_Seq_MultiChan_Dataset_Inference(eeg_file=eeg1_1,
                                                                 eog_file=eog_1,
                                                                 label_file=labels_1,
                                                                 device=device, mean_eeg_l=eeg1_m1, sd_eeg_l=eeg1_std1,
                                                                 mean_eog_l=eog_m1, sd_eog_l=eog_std1,
                                                                 sub_wise_norm=True, num_seq=num_seq,
                                                                 transform=transforms.Compose([
                                                                     transforms.ToTensor()
                                                                 ]))
        infer_data_loader = data.DataLoader(infer_dataset, batch_size=1, shuffle=False)  # 16
        len(infer_data_loader)

        test_model = torch.load(model_path, map_location=device)
        test_model.eval()  # 设置为评估模式
        pred_val_main = torch.zeros((len(infer_data_loader) + num_seq, 1, 5))  # data, output,seq pred,
        labels_val_main = torch.zeros((len(infer_data_loader) + num_seq, 1))  # data, output,seq pred,
        first = 0
        m = torch.nn.Softmax()
        with torch.no_grad():  # 确保在接下来的代码块中不会计算梯度
            test_model.eval()  # 将模型设置为评估模式，这对于推断是必要的
            for batch_val_idx, data_val in enumerate(infer_data_loader):  # 遍历数据加载器中的每个批次
                if batch_val_idx % 100 == 0: print(f"predicting ", batch_val_idx, "/", len(infer_data_loader))
                val_eeg, val_eog, val_labels = data_val  # 从批次数据中解包EEG、EOG信号和标签
                pred, _ = test_model(val_eeg.float().to(device), val_eog.float().to(device))  # 使用模型进行预测，忽略返回的第二个值

                labels_val_main[batch_val_idx:batch_val_idx + num_seq] += val_labels.squeeze().unsqueeze(
                    dim=1)  # 累加标签数据到labels_val_main数组中

                # feat_main.append(feat_list)  # 这行代码被注释掉了，它看起来像是用来存储特征的
                for ep in range(num_seq):  # 遍历每个序列
                    pred_val_main[batch_val_idx + ep] += m(pred[ep]).cpu()  # 将预测结果累加到pred_val_main数组中，m可能是一个映射函数
        pred_val_main = (pred_val_main / num_seq).squeeze()  # 计算预测的平均值
        labels_val_main = labels_val_main // num_seq  # 计算标签的平均值（这里使用整数除法）
        sens_l, spec_l, f1_l, prec_l, sens, spec, f1, prec = confusion_matrix(pred_val_main, labels_val_main, 5,
                                                                              labels_val_main.shape[0])
        g = g_mean(sens, spec)
        acc = accuracy(pred_val_main, labels_val_main)
        kap = kappa(pred_val_main, labels_val_main)
        description = f"Sleep-edf:subject {sub} day {day_}"
        print(f"Accuracy {acc}")
        print(f"Kappa {kap}")
        print(f"Macro F1 Score {f1}")
        print(f"G Mean {g}")
        print(f"Sensitivity {sens}")
        print(f"Specificity {spec}")
        print(f"Class wise F1 Score {f1_l}")
        statistic_append(result_path, description)
print("All done.")
