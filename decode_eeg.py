import re
from pathlib import Path
import scipy.io as sio
import numpy as np


class Experiment:
    def __init__(
        self,
        times,
        stim_onset,
        stim_offset,
        electrode_labels,
        electrode_xyz):

        self.times = times
        self.stim_onset = stim_onset
        self.stim_offset = stim_offset
        self.electrode_labels = electrode_labels
        self.electrode_xyz = electrode_xyz

    def load_eegdata(self,xdata=None,ydata=None,data_path=None,xdata_var=None,ydata_var=None):
        """
        Creates xdata and ydata from filepath or numpy arrays
        Inputs:
            data: path to folder containing X and Y data .mat files
                  (filenames must include 'xdata' and 'ydata' in them)
            data_var
            xdata: numpy array of EEG data of shape [subject,timepoints,electrodes,trials]
            ydata: numpy array of EEG trial labels of shape [trials]
        """

    def load_eegdata_from_mat(self,data_path,xdata_var,ydata_var,subj_name_regex):
        data_path = Path(data_path)
        self.xdata_files = list(data_path.glob('*xdata*.mat'))
        self.ydata_files = list(data_path.glob('*ydata*.mat'))

        self.nsub = len(self.xdata_files)
        self.sub_names = []
        for isub in range(self.nsub):
            filename = self.xdata_files[isub]
            self.sub_names.append(re.search(subj_name_regex,filename)[1])

            self.xdata_all[self.sub_names[isub]] = sio.loadmat(filename)[xdata_var]

    def load_eeg_from_numpy(self,xdata,ydata,unique_ids):
        self.nsub = xdata.shape[0]
        self.sub_names = range(self.nsub)
        
        self.xdata_all = {}
        self.ydata_all = {}
        for isub,sub in enumerate(self.sub_names):
            self.xdata_all[sub] = xdata[isub]
            self.ydata_all[sub] = ydata[isub]

    def load_eeg_from_dict(self,xdata,ydata):
        self.sub_names = list(xdata.keys())
        self.nsub = len(self.sub_names)
        
        self.xdata_all = xdata
        self.ydata_all = ydata
