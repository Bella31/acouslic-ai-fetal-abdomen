import subprocess
import os


if __name__ == "__main__":
    """
    Train all cross validation runs sequentially
    """
    folds = 5

    # # subopt inclusion - network 2
    # data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_using_subopt'
    # num_classes = 2
    # loss = 'CE'
    # opt_only = False
    # pos_only = True
    # mixup = True
    #
    # for fold in range(folds):
    #     print('training fold ' + str(fold))
    #     data_dir = os.path.join(data_path, str(fold))
    #     args = ("--data_dir " + data_dir + " --num_classes " + str(num_classes) + " --loss " + loss +
    #             " --opt_only " + str(opt_only) + " --pos_only " + str(pos_only) + " --apply_mixup " + str(mixup))
    #
    #     subprocess.call("python -m part1_frame_classification.scripts.train_classification " + args, shell=True)
    #
    # # subopt inclusion - network 2
    # data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_using_subopt'
    # num_classes = 2
    # loss = 'CE'
    # opt_only = False
    # pos_only = True
    # mixup = False
    #
    # for fold in range(folds):
    #     print('training fold ' + str(fold))
    #     data_dir = os.path.join(data_path, str(fold))
    #     args = ("--data_dir " + data_dir + " --num_classes " + str(num_classes) + " --loss " + loss +
    #             " --opt_only " + str(opt_only) + " --pos_only " + str(pos_only) + " --apply_mixup " + str(mixup))
    #
    #     subprocess.call("python -m part1_frame_classification.scripts.train_classification " + args, shell=True)
    #
    # # subopt inclusion - three classes
    # data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_using_subopt'
    # num_classes = 3
    # loss = 'CE'
    # opt_only = False
    # pos_only = False
    # mixup = True
    #
    # for fold in range(folds):
    #     print('training fold ' + str(fold))
    #     data_dir = os.path.join(data_path, str(fold))
    #     args = ("--data_dir " + data_dir + " --num_classes " + str(num_classes) + " --loss " + loss +
    #             " --opt_only " + str(opt_only) + " --pos_only " + str(pos_only) + " --apply_mixup " + str(mixup))
    #
    #     subprocess.call("python -m part1_frame_classification.scripts.train_classification " + args, shell=True)


    # subopt exclusion - dataset without subopt + 20 cases
    data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds_opt_no_subopt'
    num_classes = 2
    opt_only = True
    pos_only = False
    loss = 'CE'
    mixup = False

    for fold in range(folds):
        print('training fold ' + str(fold))
        data_dir = os.path.join(data_path, str(fold))
        args = ("--data_dir " + data_dir + " --num_classes " + str(num_classes) + " --loss " + loss +
                " --opt_only " + str(opt_only) + " --pos_only " + str(pos_only) + " --apply_mixup " + str(mixup))

        subprocess.call("python -m part1_frame_classification.scripts.train_classification " + args, shell=True)

