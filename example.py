"""
Example of running improved RWA (corWACFM scheme from [1]) on the distorted simulated EEG data generated using
generate_signal.m script described in [1].
"""
import mne
from RWA import RWA

# File with distorted EEG epochs generated with data/generate_signal.m
file_with_epochs = 'data/EEGLAB_distorted.set'
epochs = mne.read_epochs_eeglab(file_with_epochs)

evoked = epochs.average()
evoked.plot(window_title="Traditional average")

rwa = RWA()

# For more robust results RWA should be used only for the time window of interest
sfreq = epochs.info['sfreq']
tmin, tmax = -0.2, 0.9
baseline = round(- tmin * sfreq)
epoch_end = round(tmax * sfreq) + baseline
evoked.data[..., baseline:epoch_end] = rwa.run(epochs.get_data()[..., baseline:epoch_end])

evoked.plot(window_title="Improved RWA - time window of interest")

# Simple example using the whole epoch for averaging
evoked = epochs.average(method=rwa.run)
evoked.plot(window_title="Improved RWA - whole epoch")
