# PSD-Based Analysis and Dimensionality Reduction

This document details the first objective: EEG Data Loading, PSD Calculation, Dimensionality Reduction, and Visualization

## Data Source and Preprocessing

1. **Dataset**

    First, I decided to analyze the EEG dataset focused on lower limb motor imagery (MI) from stroke patients [1].

        [1] Liu, Y., Gui, Z., Yan, D. et al. Lower limb motor imagery EEG dataset based on the multi-paradigm and longitudinal-training of stroke patients. Sci Data 12, 314 (2025). https://doi.org/10.1038/s41597-025-04618-4

2. **Preprocessing Pipeline**

    The data files were preprocessed. Preprocessing steps included:

    1. **Artifact Channel Removal**: Exclusion of ECG and EOG channels (ECG, HEOR, HEOL, VEOU, VEOL).
    2. **Referencing**: Setting the reference to the CPz channel.
    3. **Bad Channel Handling**: Automatic removal of bad channels (amplitude deviation $> 50$ SD) followed by interpolation.
    4. **Resampling**: Downsampling the data to 250 Hz.
    5. **Filtering**: Application of a Band-Pass filter (3–35 Hz) and a Notch filter (49–51 Hz).
    6. **Epochs creating**: Segmenting the data into epochs from -3 to 7 seconds relative to events 2 and 8.
    7. **Channel Exclusion**: Further manual exclusion of 24 specific channels (FP1, FPZ, FP2, AF7, AF8, F7, F8, FT7, FT8, M1, T7, T8, M2, TP7, TP8, P7, P8, PO7, PO8, CB1, O1, OZ, O2, ECG, HEOR, HEOL, VEOU, VEOL).
    8. **Manual Cleaning**: Manual removal of bad segments and channels (followed by interpolation).
    9. **ICA Decomposition**: Independent Component Analysis (ICA) was performed.
    10. **Artifact Component Classification**: Automatic classification and removal of artifactual components.
    11. **Manual ICA Correction**: Manual inspection and removal of components.

## PSD Calculation and Initial Observations

The next step in the pipeline was the calculation of the PSD for each epoch and visualization. Significant and unexpected variability was observed in the PSD structure between subjects, and even across different experimental runs for the same subject. This high variance might indicate inconsistencies in the reported preprocessing steps across all files.

## Dimension Reduction and Clustering

Dimension reduction methods — UMAP and PCA — were applied to the PSD data. After the data was visualized. The data generally showed good cluster separation, indicating that the PSD features successfully capture distinct states.
