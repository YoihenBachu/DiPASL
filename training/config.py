# Training parameters
BATCH_SIZE = 8
EPOCHS = 30
MOMENTUM = 0.9
LEARNING_RATE = 0.001


# Other configurations
EXTENSION = ".png" #extension of the image files
NUM_CLASS = 26 #number of output classes from the model
ALPHABETS = ['A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X',
            'Y', 'Z']

MODEL_NAME = 'BYOL_CrossEntropy_AdamW_'

# Wandb 
WANDB_INIT = True