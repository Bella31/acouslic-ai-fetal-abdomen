import os
import subprocess

if __name__ == "__main__":
    """
    Evaluate 
    """
    folds = 5
    data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds'
    model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
    model_dirnames = [8,9,10,11,12]
    scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
    labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'#for test, use the original labels file that includes all frames!

    for fold in range(folds):
        model_path = os.path.join(model_dir, str(model_dirnames[fold]), 'resnet50_3class.pt')
        results_path = os.path.join(model_dir, str(model_dirnames[fold]), 'results', 'predictions.csv')
        test_scans_path = os.path.join(data_path, str(fold), 'test')
        args = ('--model ' + model_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
                + labels_path + ' --test_scans_path ' + test_scans_path)
        subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)