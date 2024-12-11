# BachelorProject

This repository serves as one of the main resources for the bachelor project conducted as part of the Bachelor Thesis titled “AI-driven Lung Analysis for Cardiovascular Disease Detection”, submitted to DTU Compute, Department of Applied Mathematics and Computer Science on December 13th 2024.

Contributors:
	•	Karoline Klan Hansen (s214638)
	•	Sofie Kodal Larsen (s214699)
	•	Jacob Seerup Kirkeby (s214596)
 
The repository contains several exploratory studies as part of the preliminary part of the bachelor project. It is organized into three main sections: DataPreprocess, PostProcess, and PreliminaryStudies.

 In 'PreliminaryStudies' you find notebooks for the preliminary studies regarding attenuation densities, gaussian mixture models (GMM), and pleural effusion.

## Repository Structure

### 1. **DataPreprocess**
In 'DataPreprocess'  scripts for preprocessing the data and extracting cropped CT data sets are found, preparing the raw CT data for Deep Learning Analysis and running nnU-net.

### 2. **PreliminaryStudies**
This folder includes Jupyter notebooks and analyses related to preliminary exploratory studies:
- **Attenuation Densities**: Investigations into the distribution and significance of attenuation densities in CT data comparing lung with and without Ground Glass Opacities (GGOs).
- **Gaussian Mixture Models (GMM)**: Applications of GMM for clustering and analysis of CT data features.
- **Pleural Effusion**: Preliminary studies on pleural effusion, focusing on its detection and characteristics in CT scans.

### 3. **PostProcess**
This folder focuses on postprocessing the outputs of the deep learning models and mapping boxplots used in the results section in the thesis.

The kidney_tools.py functions were provided by Rasmus Reinhold Paulsen, and the resample_image function were provided by Ana-Teodora Radutoiu. 
