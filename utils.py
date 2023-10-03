import torch
import os

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
 
# Dataset
TRAIN_PATH = "./data/train"
TEST_PATH = "./data/test"

BATCH_SIZE = 8
IMG_SIZE = 224
NUM_WORKERS = 1
PIN_MEMMORY = True
SHUFFLE = True

# Model
EMBED_DIM = 768
NUM_HEADS = 12
NUM_CLASSES = 3
PATCH_SIZE = 16
IMG_CHANNELS = 3
ATT_DROP = 0
EMBD_DROP = 0.1
MLP_DROP = 0.1
NUM_ENCODERS = 12
ViT_DROP = 0.1

#Train
LR = 3e-3
WEIGHT_DECAY = 0.3
EPOCHS = 10

#Save Model
SAVE_MODEL_PATH = './saved_model'

def save_model(model, model_name):

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = os.path.join(SAVE_MODEL_PATH, model_name)

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    
def load_model(model, model_name):

    # Create model load path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_load_path = os.path.join(SAVE_MODEL_PATH, model_name)

    # Load the model state_dict
    print(f"[INFO] Loading model from: {model_load_path}")
    model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    
    # Make sure to put the model in the correct mode (training or evaluation) after loading
    model.eval()  # Change to model.train() if you want to continue training
    
    return model



