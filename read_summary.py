import json
import os
import pathlib

def extract_summary_data(path_to_summary):
    '''
    given a path to a json file containing the relevant nifti filenames, and a path to the raw dataset

    extract filenames of the whitelisted nifti files from the dataset
    '''
    dice_sick_avg = []
    with open(path_to_summary, 'r') as json_file:
        summary = json.load(json_file)
    for i in range(13):
        dice_sick_avg.append(summary['metric_per_case'][i]['metrics']['2']['Dice'])
    return sum(dice_sick_avg) / len(dice_sick_avg)
    
    
summary_path = os.path.join(pathlib.Path.cwd().resolve(), 'summary.json')
avg_dice = extract_summary_data(summary_path)
print(avg_dice)