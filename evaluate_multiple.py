import os
import subprocess

if __name__ == "__main__":
    """
    Evaluate 
    """
    folds = 5
    # num_classes = 2
    # data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_using_subopt'
    # model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
    # model_dirnames = [47, 48, 49, 50, 51]
    # model2_dirnames = [76, 77, 78, 79, 80]
    # scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
    # labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'#for test, use the original labels file that includes all frames!
    #
    # for fold in range(len(model2_dirnames)):
    #     model_path = os.path.join(model_dir, str(model_dirnames[fold]), 'model.pt')
    #     model2_path = os.path.join(model_dir, str(model2_dirnames[fold]), 'model.pt')
    #     results_path = os.path.join(model_dir, str(model2_dirnames[fold]), 'results', 'predictions.csv')
    #     test_scans_lst_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'
    #   #  test_scans_path = os.path.join(data_path, str(fold), 'test')
    #     args = ('--model ' + model_path + ' --model2 ' + model2_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
    #             + labels_path + ' --test_scans_lst_path ' + test_scans_lst_path + ' --num_classes ' + str(num_classes))
    #     subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)
    #

    folds = 5
    num_classes = 2
    data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_using_subopt'
    model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
    model_dirnames = [91, 92, 93, 94, 95, 101, 102, 103, 104,105]
    scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
    labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'  # for test, use the original labels file that includes all frames!
    test_scans_lst_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'

    for ind in range(len(model_dirnames)):
        model_path = os.path.join(model_dir, str(model_dirnames[ind]), 'model.pt')
        results_path = os.path.join(model_dir, str(model_dirnames[ind]), 'results', 'predictions.csv')

        args = ('--model ' + model_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
                + labels_path + ' --test_scans_lst_path ' + test_scans_lst_path + ' --num_classes ' + str(num_classes))
        subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)