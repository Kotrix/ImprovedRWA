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

# RWA should be calculated only for the time window of interest
sfreq = epochs.info['sfreq']
tmin, tmax = -0.2, 0.9
baseline = round(- tmin * sfreq)
epoch_end = round(tmax * sfreq) + baseline
evoked.data[..., baseline:epoch_end] = RWA().run(epochs.get_data()[..., baseline:epoch_end])

evoked.plot(window_title="Improved RWA")
