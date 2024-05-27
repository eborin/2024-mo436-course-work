import sys
sys.path.append('../../')

import torch
import lightning as L

import models.deeplabv3 as dlv3
import models.barlow_twins as bt
from transforms.barlow_twins import BarlowTwinsTransforms
from data_modules.seismic import F3SeismicDataModule

# This function should load the backbone weights
def load_pretrained_backbone(pretrained_backbone_checkpoint_filename):
#    loaded_model = MyModel()`
#    loaded_model.model = AutoModel.from_pretrained("path/to/save/model")

    backbone = dlv3.DeepLabV3Backbone()
    backbone.load_state_dict(torch.load(pretrained_backbone_checkpoint_filename))
    return backbone

# This function must instantiate and configure the datamodule for the downstream task.
# You must not change this function (Check with the professor if you need to change it).
def build_downstream_datamodule() -> L.LightningDataModule:
    return F3SeismicDataModule(root_dir="../../data/", batch_size=8)

# This function must instantiate and configure the downstream model
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure it returns a Lightning model.
def build_downstream_model(backbone) -> L.LightningModule:
    return dlv3.DeepLabV3Model(backbone=backbone, num_classes=6)

# This function must instantiate and configure the lightning trainer
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure you return a Lightning trainer.
def build_lightning_trainer(SSL_technique_prefix) -> L.Trainer:
    from lightning.pytorch.callbacks import ModelCheckpoint
    # Configure the ModelCheckpoint object to save the best model 
    # according to validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'./',
        filename=f'{SSL_technique_prefix}-downstream-model',
        save_top_k=1,
        mode='min',
    )
    return L.Trainer(
        accelerator="gpu",
        max_epochs=50,
        #max_steps=10,
        logger=False,
        callbacks=[checkpoint_callback])

# This function must not be changed. 
def main(SSL_technique_prefix):

    # Load the pretrained backbone
    pretrained_backbone_checkpoint_filename = f"./{SSL_technique_prefix}_pretrained_backbone_weights.pth"
    backbone = load_pretrained_backbone(pretrained_backbone_checkpoint_filename)

    # Build the downstream model, the downstream datamodule, and the trainer
    downstream_model = build_downstream_model(backbone)
    downstream_datamodule = build_downstream_datamodule()
    lightning_trainer = build_lightning_trainer(SSL_technique_prefix)

    # Fit the pretext model using the pretext_datamodule
    lightning_trainer.fit(downstream_model, downstream_datamodule)

    # Save the downstream model    
    #output_filename = f"./{SSL_technique_prefix}_downstream_model.pth"
    #pretext_save_backbone_weights(pretext_model, output_filename)
    #print(f"Pretrained weights saved at: {output_filename}")

if __name__ == "__main__":
    SSL_technique_prefix = "BT"
    main(SSL_technique_prefix)
