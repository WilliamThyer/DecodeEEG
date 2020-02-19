from pathlib import Path
import scipy.io as sio
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import mord

class Experiment:
    def __init__(self,experiment_name,data_dir,test=False):

        self.experiment_name = experiment_name
        self.data_dir = Path(data_dir)

        self.xdata_files = list(self.data_dir.glob('*xdata*.mat'))
        self.ydata_files = list(self.data_dir.glob('*ydata*.mat'))
        if test:
            self.xdata_files = [self.xdata_files[0]]
            self.ydata_files = [self.ydata_files[0]]
        self.nsub = len(self.xdata_files)

        self.behavior_files = None
        self.artifact_idx_files = None

    def load_eeg(self,isub):
        subj_mat = sio.loadmat(self.xdata_files[0],variable_names=['xdata'])
        xdata = np.moveaxis(subj_mat['xdata'],[0,1,2],[1,2,0])

        subj_mat = sio.loadmat(self.ydata_files[0],variable_names=['ydata'])
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

class Experiment_Syncer:
    def __init__(
        self,
        experiments,
        train_test_idx
    ):
        self.experiments = experiments
        self.train_test_idx = train_test_idx
        self.experiment_names = []
        for i in range(len(experiments)):
            self.experiment_names.append(experiments[i].experiment_name)

    def _load_unique_ids(self):
        all_ids = []
        for exp in self.experiments:
            exp.unique_ids = []
            for filename in list(exp.data_dir.glob('*.txt')):
                with open(filename) as f:
                    exp.unique_ids.append(str(json.load(f)))
            all_ids.extend(exp.unique_ids)
        self.unique_ids = np.unique(all_ids)

    def _sync_unique_ids(self):
        self._load_unique_ids()
        self.matched_ids=[]
        
        for i in self.unique_ids:
            check = 0
            for exp in self.experiments:
                if i in exp.unique_ids:
                    check+=1
            if check == len(self.experiments):
                self.matched_ids.append(i)

        self.id_dict = dict.fromkeys(self.matched_ids)
        for k in self.id_dict.keys():
            self.id_dict[k] = dict.fromkeys(self.experiment_names)

        for exp in self.experiments:
            for m in self.matched_ids:
                try:
                    self.id_dict[m][exp.experiment_name] = exp.unique_ids.index(m)
                except ValueError:
                    pass
    
    def load_synced_ids(self):
        self._sync_unique_ids()

        for u in self.matched_ids:
            r = []
            for exp in self.experiments:
                r.append(exp.load_eeg(self.id_dict[u][exp.experiment_name]))
            yield r
            

class Wrangler:
    def __init__(self,
        samples,
        time_window, time_step,
        trial_average,
        n_splits,
        labels,
        electrodes = None):

        self.samples = samples
        self.sample_step = samples[1]-samples[0]
        self.time_window = time_window
        self.time_step = time_step
        self.trial_average = trial_average
        self.n_splits = n_splits
        self.labels = labels
        self.n_labels = len(labels)
        self.electrodes = electrodes

        self.cross_val = StratifiedShuffleSplit(n_splits=self.n_splits)

        self.t = samples[0:samples.shape[0]-int(time_window/self.sample_step)+1:int(time_step/self.sample_step)]
        
    def select_labels(self, xdata, ydata):
        """
        includes labels only wanted for decoding

        returns xdata and ydata with unwanted labels removed

        xdata: eeg data, shape[electrodes,timepoints,trials]
        ydata: labels, shape[trials]
        """

        restriction_idx = np.isin(ydata,self.labels)
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
    
    def setup_data(self,xdata,ydata):
        xdata,ydata = self.select_labels(xdata,ydata)
        xdata,ydata = self.balance_labels(xdata,ydata)
        xdata,ydata = self.average_trials(xdata,ydata)
        return xdata,ydata

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
    
    def roll_over_time(self, X_train_all, X_test_all):
        """
        returns one timepoint of EEG trial at a time
        """
        for self.itime, time in enumerate(self.t):
            time_window_idx = (self.samples >= time) & (self.samples < time + self.time_window)

            # Data for this time bin
            X_train = np.mean(X_train_all[...,time_window_idx],2)
            X_test = np.mean(X_test_all[...,time_window_idx],2)

            yield X_train, X_test

class Classification:
    def __init__(self,exp,wrangl,classifier=None):
        self.wrangl = wrangl
        self.n_splits = wrangl.n_splits
        self.t = wrangl.t
        self.n_labels = wrangl.n_labels
        self.exp = exp
        if classifier:
            self.classifier = classifier
        else:
            self.classifier = mord.LogisticIT()
        self.scaler = StandardScaler()

        self.acc = np.zeros((self.exp.nsub,np.size(self.t),self.n_splits))*np.nan
        self.acc_shuff = np.zeros((self.exp.nsub,np.size(self.t),self.n_splits))*np.nan
        self.conf_mat = np.zeros((self.exp.nsub,np.size(self.t),self.n_splits,self.n_labels,self.n_labels))*np.nan

    def standardize(self, X_train, X_test):
        """
        z-score each electrode across trials at this time point

        returns standardized train and test data 
        Note: this fits and transforms train data, then transforms test data with mean and std of train data!!!
        """

        # Fit scaler to X_train and transform X_train
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test

    def decode(self, X_train, X_test, y_train, y_test, isub):
        ifold = self.wrangl.ifold
        itime = self.wrangl.itime

        X_train, X_test = self.standardize(X_train, X_test)
        
        self.classifier.fit(X_train, y_train)

        self.acc[isub,itime,ifold] = self.classifier.score(X_test,y_test)
        self.acc_shuff[isub,itime,ifold] = self.classifier.score(X_test,np.random.permutation(y_test))
        self.conf_mat[isub,itime,ifold] = confusion_matrix(y_test,y_pred=self.classifier.predict(X_test))

        print(f'{round(((ifold+1)/self.n_splits)*100,1)}% ',end='\r')

