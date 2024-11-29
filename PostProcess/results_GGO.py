import json
import os
from pathlib import Path
import numpy as np
from pp_tools import load_nifti_convert_to_numpy
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

'''
To Do:

load a patient

load summary.json for cascade

get total lung volume in voxels

get # GGO voxels

output GGO / LungVolume for healthy, sick-pred, sick-GT
'''


def load_patient_get_lung_array(path_to_specific_patient):
    patient = load_nifti_convert_to_numpy(path_to_specific_patient)
    return patient

def load_summary(path_to_summary):
    with open(path_to_summary, 'r') as json_file:
        summary = json.load(json_file)
        return summary['metric_per_case']
    
def get_amount_voxels_lung(patient):
    p = np.extract(patient > -10000, patient)
    return len(p)

def get_GGO_voxels_from_summary(summary, patient, dict):
    (_,dataset, group, p_idx,_) = patient.split('_')
    GGO_gt = 0
    print(dataset,group, p_idx)
    if group == 'sick' and str(dataset) == 'Covid':
        for case in summary[7:]:
            if case['reference_file'][-10:-7]==p_idx:
                res = case['metrics']['2']
        GGO_gt = res['TP'] + res['FN']
    elif group == 'healthy' and str(dataset) == 'Covid': 
        for case in summary[:9]:
            if case['reference_file'][-10:-7]==p_idx:
                res = case['metrics']['2']
    GGO_pred = res['TP'] + res['FP']
    return GGO_gt, GGO_pred

def get_patient_dict(path_patient):
    p_dict = {}
    p_set = set()
    for p_idx in os.listdir(Path(path_patient)):
        p = int(p_idx.split('_')[3])
        p_set.add(p)
    p_set = sorted(p_set)
    for idx, val in enumerate(p_set):
        p_dict[val]=idx
    return p_dict

def write_to_json(dict):
    with open('GGO_percents_covid', 'w') as outfile:
        json.dump(dict,outfile)

def load_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)
        
def sort_result_dict(res_dict):
    hm,hgt,sm,sgt = {},{},{},{}
    for p in res_dict.keys():
        if int(p)>=10:
            hm[p]=res_dict[p][1]
            hgt[p]=res_dict[p][0]
        elif int(p)<10:
            sgt[p]=res_dict[p][0]
            sm[p]=res_dict[p][1]
    df = pd.DataFrame({'healthy_model':hm.values(),'healthy_GT':hgt.values(),'sick_model':sm.values(),'sick_GT':sgt.values()})
    return df
            

   

def make_boxplot(sorted_GGO_data):
    sns.set_theme(style="ticks", palette="pastel")
    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(x="model", y="GGO in lung (%)",
            hue="smoker", palette=["m", "g"],
            data=sorted_GGO_data)
    sns.despine(offset=10, trim=True)

        
path_patients = '/scratch/s214699/nnUNet_filer/nnUNet_raw/Dataset314_LungAnalysisCovid/imagesTr/'
path_summary_cascade = '/scratch/s214699/nnUNet_filer/nnUNet_results/Dataset314_LungAnalysisCovid/nnUNetTrainer_wandbtracker__nnUNetPlans__3d_cascade_fullres/crossval_results_folds_0_1_2_3_4/summary.json'

if __name__ == '__main__':
    # res_dict = dict()
    # patient_dict = get_patient_dict(path_patients)
    # for p_idx in os.listdir(Path(path_patients)):
    #     patient = load_patient_get_lung_array(path_patients+p_idx)
    #     lung_vol = get_amount_voxels_lung(patient)
    #     summ = load_summary(path_summary_cascade)
    #     #print(len(summ))
    #     GT, PRED = get_GGO_voxels_from_summary(summ, p_idx, patient_dict)
    #     p = p_idx.split('_')[3]
    #     res_dict[p] = [[GT/lung_vol*100 if GT>1 else 0][0],PRED/lung_vol*100]
    #     print(f'The GGO percentage for the GT of patient {p} is: {[GT/lung_vol*100 if GT>1 else 0][0]} The PRED is {PRED/lung_vol*100}')
        
    # write_to_json(res_dict)
    res_dict = load_json('GGO_percents_covid')
    sorted_data = sort_result_dict(res_dict)
    
    # # Define the DataFrame
    # data_lung = {
    #     "healthy_model": [1.080476, 0.122971, 0.734845, 0.924558, 0.015409, 0.503406, 0.150070, 2.683077, 1.648927, 1.707920, None, None, None],
    #     "healthy_GT": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None],
    #     "sick_model": [0.889374, 0.297673, 0.085086, 0.549949, 2.764871, 1.469894, 2.585738, 2.333490, 3.351203, 1.492813, 2.329664, 2.721988, 0.534492],
    #     "sick_GT": [0.007435, 0.049480, 1.654593, 0.346879, 1.657888, 2.721025, 0.905012, 10.046798, 3.890992, 0.694731, 0.242964, 0.132046, 0.032998],
    # }

    # df = pd.DataFrame(data_lung)

    # # Plotting the boxplots
    plt.figure(figsize=(12, 6))  # Set the figure size
    sns.boxplot(data=sorted_data)
    sns.despine(offset=10, trim=True)
    plt.ylabel('GGO in Lungs (%)')
    plt.axvline(x=1.5, color='red', linestyle='--', linewidth=1)
    plt.title("Ratio of GGO in lungs in (%) of total volume")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    plt.tight_layout()
    plt.savefig('boxplots')

        