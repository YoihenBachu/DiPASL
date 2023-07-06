# For all
IMG_SIZE = 256
NUM_CLASS = 26
ALPHABETS = ['A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X',
            'Y', 'Z']
EXTENSION = '.png'
DATASET_TYPE = "skeleton" # select from [skeleton, traditional]
SAVE_CSV = False 

# preprocessing parameters
FOLDER_ROOT1 = r'D:\final_year_project\dataset1'
FOLDER_ROOT2 = r'D:\final_year_project\dataset2'
BASE_LETTER = 'R'
MAXCOUNT_PER_ENV = 75

# Training parameters
BATCH_SIZE = 32
EPOCHS = 30
MOMENTUM = 0.9
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
TRAIN_BACKBONE = "resnet" # select from [resnet, mobilenet, efficientnet, xception, rexnet]
OPTIMIZER = "adam" # select from [adam, adamw, sgd]
TRANSFER = True 
DATASET_PATH = r'F:\fyp\dataset'
MODEL_SAVEPATH = r'F:\fyp'

# Prediction parameters
REAL_TIME_BACKBONE = "mobilenet" # select from [resnet, mobilenet, efficientnet, xception, rexnet]
S900_WEIGHT = 'checkpoints/DiPASL_S900/mobilenetv2_050_adamw_0.001_25.pt'
T900_WEIGHT = 'checkpoints/DiPASL_T900/mobilenetv2_050_sgd_0.01_25.pt'

# S900_WEIGHT = r"D:\final_year_project\checkpoints\DiPASL-S900\efficientnetv2_rw_m_sgd_0.001_25.pt"
# T900_WEIGHT = r"D:\final_year_project\checkpoints\DiPASL-T900\efficientnetv2_rw_m_adamw_0.0001_25.pt"

# Wandb parameters
WANDB_INIT = 'DiPASL Single Training'
WANDB_LOG = True