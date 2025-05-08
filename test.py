# import require library for preprocess
import mne
import numpy as np
from mne.channels import make_standard_montage
import matplotlib.pyplot as plt
from mne.datasets import eegbci
import scipy
import pickle
import seaborn as sns
from scipy.signal import filtfilt
import pyxdf

# import require library for classification
from sklearn.svm import SVC # SVM library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA library
from sklearn.neighbors import KNeighborsClassifier # KNN library

from sklearn.metrics import classification_report,confusion_matrix # Result representation

import pyxdf
import mne
import numpy as np

sfreq= 250
streams, header = pyxdf.load_xdf("data/sub-thai_ses-S001_task-Default_run-001_eeg.xdf") #Example Data from Lab Recoder

if streams[0]['info']['type'][0] == 'Markers': #Check
    raw_stream = streams[1]
    event_data = streams[0]
else:
    raw_stream = streams[0]
    event_data = streams[1]

raw_data = raw_stream["time_series"].T #From Steam variable this query is EEG data
channels = ['CH1','CH2','CH3','CH4','CH5','CH6','CH7','CH8'] #Set your target EEG channel name
info = mne.create_info(
    ch_names= channels,
    ch_types= ['eeg']*len(channels),
    sfreq= 250 #OpenBCI Frequency acquistion
)
raw_OpenBCI = mne.io.RawArray(raw_data, info, verbose=False)    

raw_OpenBCI.compute_psd(fmax=60).plot(picks=raw_OpenBCI.ch_names[0:5])

OpenBCI_filter = raw_OpenBCI.copy().filter(l_freq=1, h_freq=40) #band-pass function
OpenBCI_filter = OpenBCI_filter.copy().notch_filter(freqs=50) #notch filter function
OpenBCI_filter.compute_psd(fmax=60).plot(picks=raw_OpenBCI.ch_names[0:5])

event_timestamp = (np.array(event_data["time_stamps"]).T * sfreq) - (raw_stream['time_stamps'][0] * sfreq) #Timestamp when event marked
event_index = []

for i in range (len(event_data["time_series"])):
    # print(event_data["time_series"][i][0])
    event_index.append(event_data["time_series"][i][0])
    # np.append(event_index,event_data["time_series"][i][0])
np_event_index = np.array(event_index)
print(np_event_index)
events_id2 = {'101': 1,'201' : 2,'': 5}
# # Use vectorized mapping with np.vectorize
event_index = np.vectorize(events_id2.get)(event_index)

events2 = np.column_stack((np.array(event_timestamp, dtype = int),np.zeros(len(event_timestamp), dtype = int),np.array(event_index, dtype = int)))

OpenBCI_epochs = mne.Epochs(OpenBCI_filter, events2, 
        tmin= -0.5,     # init timestamp of epoch (0 means trigger timestamp same as event start)
        tmax= 2,    # final timestamp (10 means set epoch duration 10 second)
        event_id =events_id2,
        preload = True,
        event_repeated='drop',
        baseline=(-0.5, 0)
    )
import matplotlib.pyplot as plt

OpenBCI_epochs.plot(scalings = 75)
plt.show()

