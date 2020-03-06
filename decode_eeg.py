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
    def __init__(self,experiment_name ,data_dir , info_from_file = True, test = False):

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
        self.info_files = None

        if info_from_file:
            self.info = self.load_info(0)
            self.info.pop('unique_id')
            
    def load_eeg(self,isub):
        subj_mat = sio.loadmat(self.xdata_files[isub],variable_names=['xdata'])
        xdata = np.moveaxis(subj_mat['xdata'],[0,1,2],[1,2,0])

        subj_mat = sio.loadmat(self.ydata_files[isub],variable_names=['ydata'])
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
    
    def load_info(self, isub, variable_names = ['unique_id','chan_labels','chan_x','chan_y','chan_z','sampling_rate','times']):
        """ 
        loads info file that contains data about EEG file and subject
        """
        if not self.info_files:
            self.info_files = list(self.data_dir.glob('*info*.mat'))
        info_file = sio.loadmat(self.info_files[isub],variable_names=variable_names)
        
        info = {k: np.squeeze(info_file[k]) for k in variable_names}
        
        return info

class Experiment_Syncer:
    def __init__(
        self,
        experiments,
        wrangler,
        train_group
    ):
        self.experiments = experiments
        self.wrangler = wrangler
        self.train_group = train_group
        self.experiment_names = []
        for i in range(len(experiments)):
            self.experiment_names.append(experiments[i].experiment_name)

        self._load_unique_ids()
        self._find_matched_ids()

    def _load_unique_ids(self):

        all_ids = []
        for exp in self.experiments:
            exp.unique_ids = []
            for filename in list(exp.data_dir.glob('*.txt')):
                with open(filename) as f:
                    exp.unique_ids.append(str(json.load(f)))
            all_ids.extend(exp.unique_ids)

        self.matched_ids=[]
        for i in np.unique(all_ids):
            check = 0
            for exp in self.experiments:
                if i in exp.unique_ids:
                    check+=1
            if check == len(self.experiments):
                self.matched_ids.append(i)
        
    def _find_matched_ids(self):

        self.id_dict = dict.fromkeys(self.matched_ids)
        for k in self.id_dict.keys():
            self.id_dict[k] = dict.fromkeys(self.experiment_names)

        for exp in self.experiments:
            for m in self.matched_ids:
                try:
                    self.id_dict[m][exp.experiment_name] = exp.unique_ids.index(m)
                except ValueError:
                    pass
        self.nsub = len(self.matched_ids)

    def load_eeg(self,sub):
        xdata = dict.fromkeys(self.experiment_names)
        ydata = dict.fromkeys(self.experiment_names)
        for exp in self.experiments:
            xdata[exp.experiment_name],ydata[exp.experiment_name] = exp.load_eeg(self.id_dict[sub][exp.experiment_name])
        return xdata, ydata

    def select_labels(self,xdata,ydata):
        for exp_name in self.experiment_names:
            xdata[exp_name],ydata[exp_name] = self.wrangler.select_labels(xdata[exp_name],ydata[exp_name])
        return xdata,ydata

    def group_labels(self,xdata,ydata,group_dict):
        for exp_name in self.experiment_names:
            xdata[exp_name],ydata[exp_name] = self.wrangler.group_labels(xdata[exp_name],ydata[exp_name],group_dict)
        return xdata,ydata

    def balance_labels(self,xdata,ydata):
        #right now this just balances within experiment, not across
        for exp_name in self.experiment_names:
            xdata[exp_name],ydata[exp_name] = self.wrangler.balance_labels(xdata[exp_name],ydata[exp_name])
        return xdata,ydata
    
    def average_trials(self,xdata,ydata):
        for exp_name in self.experiment_names:
            xdata[exp_name],ydata[exp_name] = self.wrangler.average_trials(xdata[exp_name],ydata[exp_name])
        return xdata,ydata
    
    def setup_data(self,xdata,ydata,labels=None,group_dict=None):
        if labels:
            xdata,ydata = self.select_labels(xdata,ydata)
        if group_dict:
            xdata,ydata = self.group_labels(xdata,ydata,group_dict)
        xdata,ydata = self.balance_labels(xdata,ydata)
        xdata,ydata = self.average_trials(xdata,ydata)
        return xdata,ydata

    def group_data(self,xdata,ydata):
        xdata_train,xdata_test=None,None

        for exp_name in self.experiment_names:
            if np.isin(exp_name,self.train_group):
                if xdata_train is not None:
                    xdata_train = np.append(xdata_train,xdata[exp_name],0)
                    ydata_train = np.append(ydata_train,ydata[exp_name],0)
                elif xdata_train is None:
                    xdata_train = xdata[exp_name]
                    ydata_train = ydata[exp_name]
            else:
                if xdata_test is not None:
                    xdata_test = np.append(xdata_test,xdata[exp_name],0)
                    ydata_test = np.append(ydata_test,ydata[exp_name],0)
                elif xdata_test == None:
                    xdata_test = xdata[exp_name]
                    ydata_test = ydata[exp_name]
        return xdata_train,xdata_test,ydata_train,ydata_test

class Wrangler:
    def __init__(self,
        time_window, time_step,
        trial_average,
        n_splits,
        electrodes = None):

        self.samples = samples
        self.sample_step = samples[1]-samples[0]
        self.time_window = time_window
        self.time_step = time_step
        self.trial_average = trial_average
        self.n_splits = n_splits
        self.electrodes = electrodes

        self.cross_val = StratifiedShuffleSplit(n_splits=self.n_splits)

        self.t = samples[0:samples.shape[0]-int(time_window/self.sample_step)+1:int(time_step/self.sample_step)]
        
    def select_labels(self, xdata, ydata, labels):
        """
        includes labels only wanted for decoding

        returns xdata and ydata with unwanted labels removed

        xdata: eeg data, shape[electrodes,timepoints,trials]
        ydata: labels, shape[trials]
        """

        label_idx = np.isin(ydata,labels)
        xdata = xdata[label_idx,:,:]
        ydata = ydata[label_idx]

        return xdata, ydata

    def group_labels(self,xdata,ydata,group_dict,empty_val=9999):
        
        xdata_new = np.ones(xdata.shape)*empty_val
        ydata_new = np.ones(ydata.shape)*empty_val
        for k in group_dict.keys():
            trial_idx = np.arange(ydata.shape[0])[np.isin(ydata,group_dict[k])]
            xdata_new[trial_idx] = xdata[trial_idx]
            ydata_new[trial_idx] = k

        trial_idx = ydata_new == empty_val
        return xdata_new[~trial_idx],ydata_new[~trial_idx]
        
    def balance_labels(self,xdata,ydata,downsamp=None):
        unique_labels, counts_labels = np.unique(ydata, return_counts=True)
        if downsamp is None:
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
    
    def setup_data(self,xdata,ydata,labels=None,group_dict=None):
        if labels:
            xdata,ydata = self.select_labels(xdata,ydata,labels)
        if group_dict:
            xdata,ydata = self.group_labels(xdata,ydata,group_dict)
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

    def train_test_custom_split(self,xdata_train,xdata_test,ydata_train,ydata_test):

        cross_val_test = StratifiedShuffleSplit(n_splits=1)
        self.ifold = 0
        for train_index,_ in self.cross_val.split(xdata_train,ydata_train):
            X_train_all, y_train = xdata_train[train_index], ydata_train[train_index].astype(int)
            
            for _,test_index in cross_val_test.split(xdata_test,ydata_test):
                X_test_all, y_test = xdata_test[test_index], ydata_test[test_index].astype(int)
            
            yield X_train_all, X_test_all, y_train, y_test
            self.ifold += 1

class Classification:
    def __init__(self,wrangl,nsub,classifier=None):
        self.wrangl = wrangl
        self.n_splits = wrangl.n_splits
        self.t = wrangl.t
        self.nsub = nsub
        if classifier:
            self.classifier = classifier
        else:
            self.classifier = mord.LogisticIT()
        self.scaler = StandardScaler()

        self.acc = np.zeros((self.nsub,np.size(self.t),self.n_splits))*np.nan
        self.acc_shuff = np.zeros((self.nsub,np.size(self.t),self.n_splits))*np.nan
        self.conf_mat = np.zeros((self.nsub,np.size(self.t),self.n_splits,4,4))*np.nan

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

