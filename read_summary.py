import json
import os
import pathlib

def extract_dice(path_to_summary):
    '''
    Given a path to a nnUNet summary.json file
    
    output the average dice score for segment 2 (GGO) for the first 14 cases (all sick in LP1)
    '''
    dice_sick_avg = []
    with open(path_to_summary, 'r') as json_file:
        summary = json.load(json_file)
    for i in range(13):
        dice_sick_avg.append(summary['metric_per_case'][i]['metrics']['2']['Dice'])
    return sum(dice_sick_avg) / len(dice_sick_avg)
    
    
summary_path = os.path.join(pathlib.Path.cwd().resolve(), 'summary.json')
avg_dice = extract_dice(summary_path)
print(avg_dice)