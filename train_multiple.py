import subprocess
import os


if __name__ == "__main__":
    """
    Train all cross validation runs sequentially
    """
    folds = 5
    data_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/cross_valid_folds'
    num_classes = 2
    loss = 'CE'
    opt_only = False

    for fold in range(folds):
        print('training fold ' + str(fold))
        data_dir = os.path.join(data_path, str(fold))
        args = ("--data_dir " + data_dir + " --num_classes " + str(num_classes) + " --loss " + loss +
                " --opt_only " + str(opt_only))

        subprocess.call("python -m part1_frame_classification.scripts.train_classification " + args, shell=True)