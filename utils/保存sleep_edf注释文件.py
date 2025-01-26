import pandas as pd
import os
edf_path=r"C:\Users\Tian_Yumi\mne_data\physionet-sleep-data\SC4001EC-Hypnogram.edf"
import mne
annot = mne.read_annotations(edf_path)
annot.save("./annot_save.txt",overwrite=True)