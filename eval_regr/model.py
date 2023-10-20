import torch
import torchvision
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR

## regression network
def create_model(ch_in: int = 1):
    # model = torchvision.models.resnet18(pretrained=True)
    model = torchvision.models.resnet34(pretrained=True)
    # model = timm.create_model('resnet34', pretrained=True, in_chans=1)
    if ch_in != 3:
        model.conv1 = torch.nn.Conv2d(ch_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(512, 1)
    return model

class RegressionResnet(LightningModule):
    def __init__(
        self,
        ch_in: int = 1,
        batch_size: int = 64,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ch_in = ch_in
        self.model = create_model(ch_in=ch_in)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        source1, source2 = batch
        x1, y1 = source1
        x2, y2 = source2
        y_hat_s1 = self(x1.float())
        y_hat_s2 = self(x2.float())
        y = torch.stack([y1, y2])
        y_hat = torch.stack([y_hat_s1, y_hat_s2])
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        
    
    def evaluate(self, batch, stage=None):
        source1, source2 = batch
        x1, y1 = source1
        x2, y2 = source2
        y_hat_s1 = self(x1.float())
        y_hat_s2 = self(x2.float())
        y = torch.stack([y1, y2])
        y_hat = torch.stack([y_hat_s1, y_hat_s2])
        loss = F.mse_loss(y_hat, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="test")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            # momentum = 0.9,
            # weight_decay=1e-4,
        )

        steps_per_epoch = 45000 // self.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
