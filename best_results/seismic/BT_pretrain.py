import sys
sys.path.append('../../')

from torch import nn
import torch
import lightning as L

import models.deeplabv3 as dlv3
import models.barlow_twins as bt
from transforms.barlow_twins import BarlowTwinsTransforms
from data_modules.seismic import F3SeismicDataModule

### - Extra Code --------------------------------------------------------------------

class BT_DeepLabV3ProjectionHead(bt.BT_ProjectionHead):
    def __init__(
        self, input_dim: int = 2048, 
        hidden_dim: int = 8192, 
        output_dim: int = 8192
    ):
        super(BT_DeepLabV3ProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU()),
                (hidden_dim, hidden_dim, torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU()),
                (hidden_dim, output_dim, None, None)
            ]
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    # Process the backbone: reduce dimensionality using avgpool and flatten the 
    # tensor to match the projection head input dimensionality (2048).
    def preprocess_step(self, x):
        return self.avgpool(x).flatten(start_dim=1)

### -------------------------------------------------------------------------------

# This function must save the weights of the pretrained model
def pretext_save_backbone_weights(pretext_model, checkpoint_filename):
    print(f"Saving backbone pretrained weights at {checkpoint_filename}")
    torch.save(pretext_model.backbone.state_dict(), checkpoint_filename)

# This function must instantiate and configure the datamodule for the pretext task
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure it returns a Lightning DataModule.
def build_pretext_datamodule() -> L.LightningDataModule:
    # Build the transform object
    transform = BarlowTwinsTransforms()
    # Create the datamodule
    return F3SeismicDataModule(root_dir="../../data/",
                               batch_size=8,
                               transform=transform)

# This function must instantiate and configure the pretext model
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure it returns a Lightning model.
def build_pretext_model() -> L.LightningModule:
    # Build the backbone
    backbone = dlv3.DeepLabV3Backbone()
    # Build the projection head
    projection_head = BT_DeepLabV3ProjectionHead(input_dim=2048)
    # Build the loss function for the pretext
    loss_fn = bt.BarlowTwinsLoss(lambda_param=5e-3, gather_distributed=False)
    # Build the pretext model
    return bt.BarlowTwins(backbone=backbone, projection_head=projection_head,
                          loss_fn=loss_fn, learning_rate=0.06)

# This function must instantiate and configure the lightning trainer
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure you return a Lightning trainer.
def build_lightning_trainer() -> L.Trainer:
    return L.Trainer(
        accelerator="gpu",
        max_epochs=1,
        max_steps=30,
        enable_checkpointing=False, 
        logger=False)

# This function must not be changed. 
def main(SSL_technique_prefix):

    # Build the pretext model, the pretext datamodule, and the trainer
    pretext_model = build_pretext_model()
    pretext_datamodule = build_pretext_datamodule()
    lightning_trainer = build_lightning_trainer()

    # Fit the pretext model using the pretext_datamodule
    lightning_trainer.fit(pretext_model, pretext_datamodule)

    # Save the backbone weights
    output_filename = f"./{SSL_technique_prefix}_pretrained_backbone_weights.pth"
    pretext_save_backbone_weights(pretext_model, output_filename)

if __name__ == "__main__":
    SSL_technique_prefix = "BT"
    main(SSL_technique_prefix)
