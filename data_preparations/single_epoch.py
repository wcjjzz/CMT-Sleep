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


# ����һ���������ɳ��򣬿���ͨ��������ָ���洢���ݼ���·��������Ҫ�޸ĳ�����롿
def parse_option():
    # ���������в������������� parser�����ṩһ����������ʹ�� --help ѡ��ʱ��ʾ��
    parser = argparse.ArgumentParser('Argument for data generation')

    # ���һ����Ϊ --save_path �������в���
    # �趨��������������Ϊ�ַ�����Ĭ��ֵ��help��Ϣ��
    parser.add_argument('--save_path', type=str, default='./extract_dataset_single_epoch',
                        help='Path to store project results')

    # ������ parser �������в������н���������������洢�� opt ������
    opt = parser.parse_args()

    return opt


def signal_extract(subjects, days, channel='eeg1', filter=True, freq=[0.2, 40]):

# 1.��ʼ����Ч���ݡ�ͨ��
    ignore_data = [[13, 2], [36, 1], [39, 1], [39, 2], [52, 1], [68, 1], [68, 2], [69, 1], [69, 2], [78, 1], [78, 2],[79, 1], [79, 2]]
    all_channels = ('EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    # һ����־�������ж��Ƿ��ǵ�һ�������������
    first_sub_flag = 0

    for sub in subjects:
        for day_ in days:
            if [sub, day_] in ignore_data:
                continue
# 2.���ݻ�ȡ��ͨ������
            # fetch_data ��������������2��ֻ��1��Ԫ�ص��б�����ֵ��1��ֻ��1��Ԫ�ص��б�[data]
            # data��һ���б�����2��Ԫ�أ�����һ��EDF�ļ���·��
                # data[0] ָ��PSG�����ļ���
                # data[1] ָ��˯�߽׶α�ע�ļ���
            # ���������߱�š���¼������ţ���sleepEDF��������
            [data] = fetch_data(subjects=[sub], recording=[day_])

            # ����ͬ��ͨ������ӳ�䵽��Ӧ��������0��1��2��3�����Ա��� all_channels �б����ҵ���Ӧ��ͨ��λ�á�
            signal2idx = {"eeg1": 0, "eeg2": 1, "eog": 2, "emg": 3}

            # �õ��ų���ͨ�����б�exclude_channels
            all_channels_list = list(all_channels)
            all_channels_list.remove( all_channels[ signal2idx[channel] ] )
            exclude_channels = tuple(all_channels_list)

            # ��ָ��������ָ��������PSG�ļ�·���У���ȡ���ų���1��ͨ�����źŵ�sleep_signals
                # verbose=True���ڶ�ȡ�ļ�ʱ�����ϸ����Ϣ
                # preload=True�����ļ�����Ԥ���ص��ڴ���
            sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)

# 3.ע�Ͳü����¼�����
            # ��ָ��������ָ�������ı�ע�ļ�·���У���ȡ˯�߷��ڱ�ע��annot
            annot = mne.read_annotations(data[1])

            # ����ͬ�ķ��ڱ�ǩӳ�䵽��Ӧ��������0��1��2��3��4��5�����Ա��ڡ�
                # Movement time���˶�ʱ�䣬ָ��˯���ڼ���κ��嶯
            ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                         "Sleep stage 4": 4, "Sleep stage R": 5}
            #     # "Sleep stage ?": 5,
            #     # "Movement time": 5

            # crop�����������ǲü���Ŀ�ʼʱ�䡢����ʱ��
                # ��ȥ30���Ӻ�ʼʱ����Ϊ����, ������ʹ�õĿ⺯�������С������Ϊ0������
            # �ü���Ϊ��ȷ����ע��������
            annot.crop(annot[1]['onset'] - 30 * 60,
                              annot[-2]['onset'] + 30 * 60)

            # ���ü���ı�ע��Ϣ���뵽�źŶ���sleep_signals��
                # emit_warning=False����ʾ������ע��ʱ������������Ϣ
            sleep_signals.set_annotations(annot, emit_warning=False)

            # events_from_annotations�������������źŶ����¼�id��ԭʼ��ע��ӳ���ϵ���¼�Ԫ�ص�ʱ��
            # �����źŶ���ע����Ϣ�����¼�����events
            events, _ = mne.events_from_annotations(
                sleep_signals, event_id=ann2label, chunk_duration=30.)

# 4.�źŹ���

            # �������Ϊ��Ҫ�˲�
                # filter�������������źŵĸ����˲����������趨�����½�ֹƵ��
            if filter == True:
                sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

# 5.���� Epoch ����ȡ����
#���źŶ���sleep_signals�������¼�����events�����ֳ�epoch��������

            # Epoches��������ԭʼ�źŶ����и�Ϊepoch�������飬������ԭʼ�źš��¼����顢�¼�id���ע��ӳ�䡢
                # ÿ��epoch����¼���ʼʱ���ʱ����㡾tmin = 0����
                # ÿ��epoch����¼���ʼʱ���ʱ���յ㡾ÿ��epoch��ʱ��Ϊ 30 ���ȥһ�������㣬����߽����⡿��
                # �Ƿ���źŽ��л���У��
                # �Ƿ�Ԥ���ص��ڴ�
                # ����ȱʧ���¼����������⣬�ᷢ������
            tmax = 30. - 1. / sleep_signals.info['sfreq']
            epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                                     event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                                     on_missing='warn')

            # epoch�źŶ����б��¼�id�б�
            sig_epochs = []
            label_epochs = []

            # ÿ��epoch���źž�ֵ�ͱ�׼��
            mean_epochs = []
            std_epochs = []

            signal_mean = np.mean(np.array([epochs_data]))
            signal_std = np.std(np.array([epochs_data]))

# 6.��ǩ��������ݴ洢
# һ�δ���ȫ��epoch���õ�sig_epochs�б�label_epochs�б���״��Ϊ (1, n_epoches)��
            for ep in range(len(epochs_data)):
                # ��epoch�е�ÿ���������ź�������ȡΪsig���顾��״Ϊ (1, n_samples)��
                    # sig_epochs �б�Ԫ��Ϊÿ��epoch���ź��������顾��״Ϊ (1, n_epoches)��
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

# 7.ƴ������
            # �б�ת��������
            sig_epochs = np.array(sig_epochs)
            mean_epochs = np.array(mean_epochs)
            std_epochs = np.array(std_epochs)
            label_epochs = np.array(label_epochs)

            # ���������ǵ�һ�������ߣ�Ҫ��ʼ��4���洢����
            if first_sub_flag == 0:
                main_ext_raw_data = sig_epochs    # �����������ߵ��и��epoches���ź����ݡ�
                main_labels = label_epochs    # ���������ߵı�ǩ����
                main_sub_len = np.array([len(epochs_data)])    # ÿ�������ߵ� epoch ����
                main_mean = mean_epochs    # ���������ߵ��źž�ֵ����
                main_std = std_epochs      # ���������ߵ��źű�׼������
                first_sub_flag = 1
            else:
                main_ext_raw_data = np.concatenate((main_ext_raw_data, sig_epochs), axis=0)
                main_labels = np.concatenate((main_labels, label_epochs), axis=0)
                main_sub_len = np.concatenate((main_sub_len, np.array([len(epochs_data)])), axis=0)
                main_mean = np.concatenate((main_mean, mean_epochs), axis=0)
                main_std = np.concatenate((main_std, std_epochs), axis=0)

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std


def main():
# 1.�ӿ���̨��ȡ������������ɵ����ݼ��ı���·����
    args = parse_option()

# 2.ʹ��KFold���������������Ϊ5�顾5�۽�����֤����ȡ1����Ϊ��֤����4����Ϊѵ������
    from sklearn.model_selection import KFold
    days = np.arange(1, 3)
    subjects = np.arange(0, 83)    # 0-82�������ߵı��λ������������߱��ѱ����ڽ�����֤���ͱ��Ϊ1��
    print(f"Subjects : {subjects}")
    print(f"Days : {days}")

    fivefold_list = []     # �����洢ÿһ�����ݵ���������
    # ����һ�� KFold ���󣬲��������ֳɼ��ۣ�����ǰ�Ƿ���ҡ�������������ӡ�ȷ��ÿ�λ�����ͬ��
    kf = KFold(n_splits=5, shuffle=True  # 5, 2
               , random_state=2
               )

    #�������߷ֳ�5�飬���洢ÿһ��������߱��
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

# 3.���Ѿ�������������֤�������ߣ�subjects���λ���Ϊ1
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

# 4.��ȡ���иָ��ͨ��������������źš�eeg1��eog������sub_1��sub_2��sub_3��sub_4��sub_5��
    # �Ե�һ�������ߵ�ԭʼ���ݣ���Ԥ���������أ�epoch���顢��ǩ���顢epoch�������źž�ֵ���źű�׼�
    eeg1_1, labels_1, len_1, eeg1_m1, eeg1_std1 = signal_extract(sub_1, days, channel='eeg1', filter=True,
                                                                 freq=[0.2, 40])
    print(f"Train data shape : {eeg1_1.shape}, Train label shape : {labels_1.shape}")


# 5.���õ������ݼ��ĸ�����Ϣ�����飬����� HDF5 �ļ���
    hf = h5py.File(f'{save_path}/x1.h5', 'w')      # epoch���顾x1��
    hf.create_dataset('data', data=eeg1_1)
    hf.close()
    hf = h5py.File(f'{save_path}/y1.h5', 'w')      # ��ǩ���顾y1��
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
