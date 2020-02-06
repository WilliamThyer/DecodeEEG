import re
from pathlib import Path
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class Experiment:
    def __init__(self,experiment_name,data_dir):

        self.experiment_name = experiment_name
        self.data_dir = Path(data_dir)

        self.xdata_files = list(self.data_dir.glob('*xdata*.mat'))
        self.ydata_files = list(self.data_dir.glob('*ydata*.mat'))
        self.nsub = len(self.xdata_files)

        self.behavior_files = None
        self.artifact_idx_files = None

    def load_eeg(self,xdata_filepath,ydata_filepath):
        subj_mat = sio.loadmat(xdata_filepath,variable_names=['xdata'])
        xdata = np.moveaxis(subj_mat['xdata'],[0,1,2],[1,2,0])

        subj_mat = sio.loadmat(ydata_filepath,variable_names=['ydata'])
        ydata = np.squeeze(subj_mat['ydata'])

        return xdata, ydata

    def load_behavior(self, isub, remove_artifact_trials=True):
        """
        returns behavior from csv as dictionary
        
        remove_artifact_trials will remove all behavior trials that were excluded from EEG data due to artifacts
        """
        if not self.behavior_files:
            self.behavior_files = list(self.data_dir.glob('*.csv'))
        behavior = pd.read_csv(self.behavior_files[isub]).to_dict('list')

        if remove_artifact_trials:
            artifact_idx = self.load_artifact_idx(isub)
            for k in behavior.keys():
                behavior[k] = np.array(behavior[k])[artifact_idx]
        else:
            for k in behavior.keys():
                behavior[k] = np.array(behavior[k])

        return behavior
        
    def load_artifact_idx(self, isub):
        """
        returns artifact index from EEG artifact rejection
        
        useful for removing behavior trials not included in EEG data
        """
        if not self.artifact_idx_files:
            self.artifact_idx_files = list(self.data_dir.glob('*idx*.mat'))
        artifact_idx = np.squeeze(sio.loadmat(self.artifact_idx_files[isub])['filt_idx']==1)

        return artifact_idx


class Wrangler:
    def __init__(self,
        samples, sampling_rate,
        time_window, time_step,
        trial_average,
        n_splits,
        electrodes = None, include_labels = None):

        self.samples = samples
        self.sample_rate = sampling_rate
        self.sample_step = sampling_rate/1000
        self.time_window = time_window
        self.time_step = time_step
        self.trial_average = trial_average
        self.n_splits = n_splits
        self.electrodes = electrodes
        self.include_labels = include_labels

        self.cross_val = StratifiedShuffleSplit(n_splits=self.n_splits)

        self.t = self.times[0:self.num_times-int(self.time_bin_size/2)+1:int(self.time_bin_offset/2)]
        
    def select_labels(self, xdata, ydata):
        """
        includes labels only wanted for decoding

        returns xdata and ydata with unwanted labels removed

        xdata: eeg data, shape[electrodes,timepoints,trials]
        ydata: labels, shape[trials]
        """
        if self.include_labels:
            restriction_idx = np.isin(ydata,self.include_labels)
            xdata = xdata[restriction_idx,:,:]
            ydata = ydata[restriction_idx]

        return xdata, ydata

    def balance_labels(self,xdata,ydata):
        unique_labels, counts_labels = np.unique(ydata, return_counts=True)
        downsamp = min(counts_labels)
        
        label_idx=[]
        for label in unique_labels:
            label_idx = np.append(label_idx,np.random.choice(np.arange(len(ydata))[ydata == label],downsamp,replace=False))
        
        xdata = xdata[label_idx.astype(int),:,:]
        ydata = ydata[label_idx.astype(int)]

        return xdata,ydata
    
    def average_trials(self,xdata,ydata):
        if self.trial_average:
            unique_labels, counts_labels = np.unique(ydata, return_counts=True)
            min_count = np.floor(min(counts_labels)/self.trial_average)*self.trial_average
            nbin = int(min_count/self.trial_average)
            trial_groups = np.tile(np.arange(nbin),self.trial_average)

            xdata_new = np.zeros((nbin*len(unique_labels),xdata.shape[1],xdata.shape[2]))
            count = 0
            for ilabel in unique_labels:
                for igroup in np.unique(trial_groups):
                    xdata_new[count] = np.mean(xdata[ydata==ilabel][:int(min_count)][trial_groups==igroup],axis=0)
                    count += 1
            ydata_new = np.repeat(unique_labels,nbin)
            return xdata_new, ydata_new
        else: return xdata,ydata

    def train_test_split(self, xdata, ydata):
        """
        returns xtrain and xtest data and respective labels
        """
        self.ifold = 0
        for train_index, test_index in self.cross_val.split(xdata[:,0,0], ydata):
        
            X_train_all, X_test_all = xdata[train_index], xdata[test_index]
            y_train, y_test = ydata[train_index].astype(int), ydata[test_index].astype(int)

            yield X_train_all, X_test_all, y_train, y_test
            self.ifold += 1
