# MI-EEG-Transformer: Transformer for Motor Imagery EEG Classification

This repository contains the official code for the paper **"Cybernetics-Enhanced Transformer for Motor Imagery EEG Classification"** (Submitted to International Journal of Machine Learning and Cybernetics).

## Overview
The code implements a hybrid model combining Transformer self-attention and cybernetics-based feedback, achieving 92.3% classification accuracy on the BCI Competition IV Dataset 2a.

## Environment Setup
1. Clone this repo:
   ```bash
   git clone https://github.com/saihanh0820@gmail.com/MI-EEG-Transformer.git
   cd MI-EEG-Transformer
python==3.9.12
torch==2.0.1
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
mne==1.3.0
transformers==4.30.2
tqdm==4.65.0
