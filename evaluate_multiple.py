import json
import os
import subprocess

if __name__ == "__main__":
    """
    Evaluate 
    """
    folds = 5
    num_classes = 2
    data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_using_subopt'
    model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
    model_dirnames = [196, 197, 198, 199, 200]
    model2_dirnames = [201, 202, 203, 204, 205]
    scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
    labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'  # for test, use the original labels file that includes all frames!
   # test_scans_lst_path = None
    test_scans_path = None
    threshold = 0
    top5 = 'false'
    test_scans_lst_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'
    for fold in range(len(model2_dirnames)):
        model_path = os.path.join(model_dir, str(model_dirnames[fold]), 'model.pt')
        model2_path = os.path.join(model_dir, str(model2_dirnames[fold]), 'model.pt')
        # with open(os.path.join(os.path.dirname(model2_path), 'config.json'), 'r') as f:
        #     config = json.load(f)
        # test_scans_path = os.path.join(config['data_dir'], 'test')
        results_path = os.path.join(model_dir, str(model2_dirnames[fold]), 'results',
                                    f'predictions_{model2_dirnames[fold]}_subopt_all_rel.xlsx')
        args = (
                    '--model ' + model_path + ' --model2 ' + model2_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
                    + labels_path + ' --test_scans_lst_path ' + str(test_scans_lst_path) + ' --test_scans_path ' +
                    str(test_scans_path) + ' --num_classes ' + str(num_classes) + ' --threshold ' + str(threshold) +
                    ' --top5 ' + top5)
        print(args)
        subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)

    folds = 5
    num_classes = 2
    data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_using_subopt'
    model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
    model_dirnames = [196, 197, 198, 199, 200]
    model2_dirnames = [201, 202, 203, 204, 205]
    scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
    labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'#for test, use the original labels file that includes all frames!
    test_scans_lst_path = None
    test_scans_path = None
    threshold = 0
    top5 = 'false'
  #  test_scans_lst_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'
    for fold in range(len(model2_dirnames)):
        model_path = os.path.join(model_dir, str(model_dirnames[fold]), 'model.pt')
        model2_path = os.path.join(model_dir, str(model2_dirnames[fold]), 'model.pt')
        with open(os.path.join(os.path.dirname(model2_path), 'config.json'), 'r') as f:
            config = json.load(f)
        test_scans_path = os.path.join(config['data_dir'], 'test')
        results_path = os.path.join(model_dir, str(model2_dirnames[fold]), 'results', f'predictions_{model2_dirnames[fold]}_all_rel.xlsx')
      #  test_scans_path = os.path.join(data_path, str(fold), 'test')
        args = (
                '--model ' + model_path + ' --model2 ' + model2_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
                + labels_path + ' --test_scans_lst_path ' + str(test_scans_lst_path) + ' --test_scans_path ' +
                str(test_scans_path) + ' --num_classes ' + str(num_classes) + ' --threshold ' + str(threshold) +
                ' --top5 ' + top5)
        print(args)
        subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)

    folds = 5
    num_classes = 2
    data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_using_subopt'
    model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
    model_dirnames = [196, 197, 198, 199, 200]
    model2_dirnames = [201, 202, 203, 204, 205]
    scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
    labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'  # for test, use the original labels file that includes all frames!
   # test_scans_lst_path = None
    test_scans_path = None
    threshold = 0
    top5 = 'true'
    test_scans_lst_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'
    for fold in range(len(model2_dirnames)):
        model_path = os.path.join(model_dir, str(model_dirnames[fold]), 'model.pt')
        model2_path = os.path.join(model_dir, str(model2_dirnames[fold]), 'model.pt')
        # with open(os.path.join(os.path.dirname(model2_path), 'config.json'), 'r') as f:
        #     config = json.load(f)
        # test_scans_path = os.path.join(config['data_dir'], 'test')
        results_path = os.path.join(model_dir, str(model2_dirnames[fold]), 'results',
                                    f'predictions_{model2_dirnames[fold]}_subopt.xlsx')
        args = (
                    '--model ' + model_path + ' --model2 ' + model2_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
                    + labels_path + ' --test_scans_lst_path ' + str(test_scans_lst_path) + ' --test_scans_path ' +
                    str(test_scans_path) + ' --num_classes ' + str(num_classes) + ' --threshold ' + str(threshold) +
                    ' --top5 ' + top5)
        print(args)
        subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)

    folds = 5
    num_classes = 2
    data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_using_subopt'
    model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
    model_dirnames = [196, 197, 198, 199, 200]
    model2_dirnames = [201, 202, 203, 204, 205]
    scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
    labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'#for test, use the original labels file that includes all frames!
    test_scans_lst_path = None
    test_scans_path = None
    threshold = 0
    top5 = 'true'
  #  test_scans_lst_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'
    for fold in range(len(model2_dirnames)):
        model_path = os.path.join(model_dir, str(model_dirnames[fold]), 'model.pt')
        model2_path = os.path.join(model_dir, str(model2_dirnames[fold]), 'model.pt')
        with open(os.path.join(os.path.dirname(model2_path), 'config.json'), 'r') as f:
            config = json.load(f)
        test_scans_path = os.path.join(config['data_dir'], 'test')
        results_path = os.path.join(model_dir, str(model2_dirnames[fold]), 'results', f'predictions_{model2_dirnames[fold]}.xlsx')
      #  test_scans_path = os.path.join(data_path, str(fold), 'test')
        args = (
                '--model ' + model_path + ' --model2 ' + model2_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
                + labels_path + ' --test_scans_lst_path ' + str(test_scans_lst_path) + ' --test_scans_path ' +
                str(test_scans_path) + ' --num_classes ' + str(num_classes) + ' --threshold ' + str(threshold) +
                ' --top5 ' + top5)
        print(args)
        subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)
# folds = 5
# num_classes = 2
# data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_no_subopt'
# model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
# model_dirnames = [168, 169]
# split_path = [3, 4]
# scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
# labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'  # for test, use the original labels file that includes all frames!
# #test_scans_lst_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'
# #test_scans_path = None
# test_scans_lst_path = None
# for ind in range(len(model_dirnames)):
#     model_path = os.path.join(model_dir, str(model_dirnames[ind]), 'model.pt')
#     with open(os.path.join(os.path.dirname(model_path), 'config.json'), 'r') as f:
#         config = json.load(f)
#     test_scans_path = os.path.join(config['data_dir'], 'test')
#     results_path = os.path.join(model_dir, str(model_dirnames[ind]), 'results', f'predictions_{model_dirnames[ind]}.xlsx')
#
#     args = ('--model ' + model_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
#             + labels_path + ' --test_scans_lst_path ' + str(test_scans_lst_path) + ' --test_scans_path ' +
#             str(test_scans_path) +' --num_classes ' + str(num_classes))
#     subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)
#
# folds = 5
# num_classes = 2
# data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_no_subopt'
# model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
# model_dirnames = [168, 169]
# split_path = [3, 4]
# scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
# labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'  # for test, use the original labels file that includes all frames!
# test_scans_lst_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'
# test_scans_path = None
# #  test_scans_lst_path = None
# for ind in range(len(model_dirnames)):
#     model_path = os.path.join(model_dir, str(model_dirnames[ind]), 'model.pt')
#     # with open(os.path.join(os.path.dirname(model_path), 'config.json'), 'r') as f:
#     #     config = json.load(f)
#     # test_scans_path = os.path.join(config['data_dir'], 'test')
#     results_path = os.path.join(model_dir, str(model_dirnames[ind]), 'results', f'predictions_subopt_{model_dirnames[ind]}.xlsx')
#
#     args = ('--model ' + model_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
#             + labels_path + ' --test_scans_lst_path ' + str(test_scans_lst_path) + ' --test_scans_path ' +
#             str(test_scans_path) +' --num_classes ' + str(num_classes))
#     subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)


#
# folds = 5
# num_classes = 2
# data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_no_subopt'
# model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
# model_dirnames = [170, 171, 172, 173, 174]
# split_path = [0, 1, 2, 3, 4]
# scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
# labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'  # for test, use the original labels file that includes all frames!
# test_scans_lst_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'
# test_scans_path = None
# #  test_scans_lst_path = None
# for fold in range(len(model_dirnames)):
#     model_path = os.path.join(model_dir, str(model_dirnames[fold]), 'model.pt')
#     # with open(os.path.join(os.path.dirname(model_path), 'config.json'), 'r') as f:
#     #     config = json.load(f)
#     # test_scans_path = os.path.join(config['data_dir'], 'test')
#     results_path = os.path.join(model_dir, str(model_dirnames[fold]), 'results',
#                                 f'predictions_{model_dirnames[fold]}_subopt.xlsx')
#
#     args = ('--model ' + model_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
#             + labels_path + ' --test_scans_lst_path ' + str(test_scans_lst_path) + ' --test_scans_path ' +
#             str(test_scans_path) +' --num_classes ' + str(num_classes))
#     subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)
#
# folds = 5
# num_classes = 2
# data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_no_subopt'
# model_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
# model_dirnames = [170, 171, 172, 173, 174]
# split_path = [0, 1, 2, 3, 4]
# scan_dir = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
# labels_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/labels.csv'  # for test, use the original labels file that includes all frames!
# #test_scans_lst_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/subopt_cases.txt'
# #test_scans_path = None
# test_scans_lst_path = None
# for fold in range(len(model_dirnames)):
#     model_path = os.path.join(model_dir, str(model_dirnames[fold]), 'model.pt')
#     with open(os.path.join(os.path.dirname(model_path), 'config.json'), 'r') as f:
#         config = json.load(f)
#     test_scans_path = os.path.join(config['data_dir'], 'test')
#     results_path = os.path.join(model_dir, str(model_dirnames[fold]), 'results', f'predictions_{model_dirnames[fold]}.xlsx')
#
#     args = ('--model ' + model_path + ' --scan_dir ' + scan_dir + ' --output ' + results_path + ' --labels_path '
#             + labels_path + ' --test_scans_lst_path ' + str(test_scans_lst_path) + ' --test_scans_path ' +
#             str(test_scans_path) +' --num_classes ' + str(num_classes))
#     subprocess.call("python -m part1_frame_classification.scripts.evaluate_frame_selection " + args, shell=True)
