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

# preprocessing part
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
BACKBONE = "resnet18"
OPTIMIZER = "adam"
TRANSFER = True

# Other configurations
MODEL_NAME = 'BYOL_CrossEntropy_AdamW_'
SAVE_CSV = False

# Wandb 
WANDB_INIT = 'DiPASL Single Training'
WANDB_LOG = True