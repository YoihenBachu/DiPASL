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
DATASET_TYPE = "line_plotted" # select from [line_plotted, traditional]

# preprocessing parameters
FOLDER_ROOT1 = r'D:\final_year_project\dataset1'
FOLDER_ROOT2 = r'D:\final_year_project\dataset2'
BASE_LETTER = 'R'
MAXCOUNT_PER_ENV = 75

# Training parameters
BATCH_SIZE = 64
EPOCHS = 30
MOMENTUM = 0.9
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
TRAIN_BACKBONE = "resnet18"
OPTIMIZER = "adam"
TRANSFER = True

# Prediction parameters
PREDICT_BACKBONE = "efficientnetv2_rw_m.agc_in1k"
WEIGHT_PATH = r'F:\fyp\new_weights\weight1\efficientnetv2_rw_m.agc_in1k_adamw_0.001_25.pt'

# Other configurations
SAVE_CSV = False

# Wandb parameters
WANDB_INIT = 'DiPASL Single Training'
WANDB_LOG = True