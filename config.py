##################################################
# Default Config
##################################################
# Directory
ckpt_dir = "./checkpoints/"  # saving directory of .ckpt models
result_pth = "./results/"
directories = [ckpt_dir, result_pth]

# Logging Config
logg_path = "./log/"

# Dataset/Path Config
img_folder = "/home/mbl/Yiyuan/CV_hw2/data/train/"  # training dataset path
test_img_folder = "/home/mbl/Yiyuan/CV_hw2/data/test/"  # testing images
annotation_file = "/home/mbl/Yiyuan/CV_hw2/data/dummy.pkl"  # training label

split = 0.01  # percentage of validation set
workers = 0  # number of Dataloader workers

# Detection Task Config
num_classes = 11  # 1 class (person) + background

##################################################
# Training Config
##################################################
device = "cuda:0"
model_name = "fastRCNN_"
log_name = "train.log"  # Beta


# Hyper-parameters Config
epochs = 50  # number of epochs
batch_size = 8  # batch size
learning_rate = 1e-3  # initial learning rate

##################################################
# Eval Config
##################################################
parameter_resnext50_32x8d = 'fastRCNN_op_SGD_lr_cylr_resnext50_32x8d_dataaug_shear_rotate_crop_noFlip__zoomin_loss8'
parameter_resnet101 = 'fastRCNN_op_SGD_lr_cylr_resnet101_dataaug_shear_rotate_crop_noFlip__zoomin_loss55'
resnet101_model_link = 'https://drive.google.com/file/d/1lYDxtcELuzWSlnekhOOjHC04EHt55SM2/view?usp=sharing'
resNext50_32x8d_model_link = 'https://drive.google.com/file/d/1MigJP5obIAY-8NeA3UwxinnewEC1RBs2/view?usp=sharing'
prediction_file_name = model_name + ".json"  # saving prediction result
