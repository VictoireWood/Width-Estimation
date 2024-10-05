
import os
import argparse
from datetime import datetime

# from he_database_generate import patches_save_root_dir    # EDIT

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # EDIT
    # parser.add_argument("--H", type=int, default=100)   # NOTE: 按照高度分组

    # Training parameters
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("-ipe", "--iterations_per_epoch", type=int, default=2000, help="_")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="_")
    parser.add_argument("--scheduler_patience", type=int, default=10, help="_")
    parser.add_argument("--epochs_num", type=int, default=500, help="_")
    # parser.add_argument("--train_resize", type=int, default=(224, 224), help="_") # ANCHOR
    # parser.add_argument("--train_resize", type=tuple, default=(360, 480), help="_") # REVIEW version 1
    parser.add_argument("--train_resize", type=int, default=(222, 296), help="_")   # REVIEW    如果用DINOv2，就改成210*280
    # parser.add_argument("--test_resize", type=int, default=256, help="_")           # ANCHOR
    parser.add_argument("--test_resize", type=int, default=222, help="_")           # REVIEW

    parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--classifier_lr", type=float, default=0.01, help="_")
    parser.add_argument("-bb", "--backbone", type=str, default="EfficientNet_B0",
                        # choices=["EfficientNet_B0", "EfficientNet_B5", "EfficientNet_B7"],    # ANCHOR 原始
                        choices=["EfficientNet_B0", "EfficientNet_B5", "EfficientNet_B7","EfficientNet_V2_M"],  # REVIEW 邵星雨改
                        help="_")
    # EDIT
    # Test parameters
    parser.add_argument('--threshold', type=int, default=None, help="验证是否成功召回的可容许偏差的距离，单位为米")    # REVIEW M自适应的话可以不设置


    # Init parameters
    parser.add_argument("--resume_train", type=str, default=None, help="path with *_ckpt.pth checkpoint")
    parser.add_argument("--resume_model", type=str, default=None, help="path with *_model.pth model")

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
    parser.add_argument("--num_workers", type=int, default=16, help="_")

    # Paths parameters
    parser.add_argument("--exp_name", type=str, default="default",
                        help="name of experiment. The logs will be saved in a folder with such name")
    parser.add_argument("--dataset_name", type=str, default="sf_xl",
                        choices=["sf_xl", "tokyo247", "pitts30k", "pitts250k","QingDao_Flight"], help="_")   # REVIEW
                        # choices=["sf_xl", "tokyo247", "pitts30k", "pitts250k"], help="_") # ANCHOR
    parser.add_argument("--train_set_path", type=str, default=None,
                        help="path to folder of train set")
    parser.add_argument("--val_set_path", type=str, default=None,
                        help="path to folder of val set")
    parser.add_argument("--test_set_path", type=str, default=None,
                        help="path to folder of test set")
    
    args = parser.parse_args()

    # EDIT
    if args.exp_name == "default":
        args.exp_name = f'WE-{args.backbone}-{args.classifier_type}-{args.N}-{args.M}-h{flight_heights[0]}~{flight_heights[-1]}'


    args.save_dir = os.path.join("logs", args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    return args

