from base import train_model
from seismic_model import Seismic_PreTrain_Config
from datetime import datetime

# Configuration dictionaries

# Model configuration
#  Passed as argument to Seismic_PreTrain_Config()->model()
model_kwargs = {
    "learning_rate": 0.06,
    "bt_loss_args" : {
        "lambda_param": 5e-3,
        "gather_distributed": False
    }
}

# Data module configuration
#  Passed as argument to Seismic_PreTrain_Config()->data_module()
data_module_kwargs = {
    "root_dir": "data/",
    "batch_size": 8
}

# Trainer configuration
#  Passed as argument to Seismic_PreTrain_Config()->trainer()
trainer_kwargs = {
    "save_dir": "logs/pretrain/",
    "name": "barlow_twins_seismic",
    "version": datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
    "epochs": 3,
    "accelerator": "gpu",
    "monitor": None,
    "mode": "min",
}

def main():
    config = Seismic_PreTrain_Config()

    # This method will employ the config object to create the model, the data_module, and the trainer. 
    # Then, it will invoke the trainer.fit() method to train the model.
    # You must not change the train_code() function nor any code at the base.py file.
    train_model(config, model_kwargs, data_module_kwargs, trainer_kwargs)

if __name__ == "__main__":
    main()