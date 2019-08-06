# Tom-Matthews

This project involved the use of supervised machine learning and EEG based drowsiness correlates to predict reaction times in driving simulation datasets.

For any questions email: tomukmatthews@gmail.com

_**Requirements:**_

**Software:**
* Matlab (used R2018b) - signal processing toolbox, statistics toolbox, machine learning toolbox
* EEGlab (used eeglab14_1_2b)
* fieldtrip (used fieldtrip-20190120)
* BLINKER - an EEGlab plugin to detect and extract the blinks from the frontal electrodes

BLINKER link: http://vislab.github.io/EEG-Blinks/#Plugin
The blinkProperties structure must be run on each dataset and saved for use in 'classify_computeBlinksfts.m'

**Data:**
* Epoched data (minimum ~5s for most of the scripts)
* Sampled at 256 Hz

(Rest of pre-processing steps used are in the project pdf)

Note, an alteration to the EEGlab function 'pop_epoch.m' is necessary for the script 'classification_features_epch.m' to run, the altered script is included in this repository.

