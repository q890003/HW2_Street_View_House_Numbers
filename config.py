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
prediction_file_name = model_name + ".json"  # saving prediction result
