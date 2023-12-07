import torch
import torchvision
# from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR

## regression network

def create_model_res50_old(ch_in: int = 1, ncls: int = None, task: str = "regression"):
    if task == "classification":
        assert ncls is not None
    model = torchvision.models.resnet50(pretrained=True)
    if task == "regression":
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        if ncls == 2:
            model.fc = nn.Linear(model.fc.in_features, 1)
        else:
            model.fc = nn.Linear(model.fc.in_features, ncls)
    if ch_in != 3:
        model.conv1 = nn.Conv2d(ch_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def create_model_res101_old(ch_in: int = 1, ncls: int = None, task: str = "regression"):
    if task == "classification":
        assert ncls is not None
    model = torchvision.models.resnet101(pretrained=True)
    if task == "regression":
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        if ncls == 2:
            model.fc = nn.Linear(model.fc.in_features, 1)
        else:
            model.fc = nn.Linear(model.fc.in_features, ncls)
    if ch_in != 3:
        model.conv1 = nn.Conv2d(ch_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model
def create_model_vgg19(ch_in: int = 1, ncls: int = None, task: str = "regression"):
    if task == "classification":
        assert ncls is not None
    model = torchvision.models.vgg19(pretrained=False, num_classes=ncls)
    # vgg_features = nn.Sequential(*list(model.children())[:-1])
    # fc = nn.Linear(model.classifier[-1].in_features, ncls)
    # model = nn.Sequential(vgg_features, nn.Flatten(), fc)
    return model

def create_model_inceptionv3(ch_in: int = 1, ncls: int = None, task: str = "regression"):
    if task == "classification":
        assert ncls is not None
    model = torchvision.models.inception_v3(pretrained=True)

    if task == "regression":
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        if ncls == 2:
            model.fc = nn.Linear(model.fc.in_features, 1)
        else:
            model.fc = nn.Linear(model.fc.in_features, ncls)
    model.aux_logits = False
    # if ch_in != 3:
    #     resnet_features = nn.Sequential(*list(model.children())[1:-1])
    #     conv1 = nn.Conv2d(ch_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #     model = nn.Sequential(conv1, resnet_features, nn.Flatten(), fc)
    return model

def create_model_res18(ch_in: int = 1, ncls: int = None, task: str = "regression"):
    if task == "classification":
        assert ncls is not None
    model = torchvision.models.resnet18(pretrained=True)

    if task == "regression":
        fc = nn.Linear(model.fc.in_features, 1)
    else:
        if ncls == 2:
            fc = nn.Linear(model.fc.in_features, 1)
        else:
            fc = nn.Linear(model.fc.in_features, ncls)
    if ch_in != 3:
        resnet_features = nn.Sequential(*list(model.children())[1:-1])
        conv1 = nn.Conv2d(ch_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = nn.Sequential(conv1, resnet_features, nn.Flatten(), fc)
    else:
        resnet_features = nn.Sequential(*list(model.children())[:-1])
        model = nn.Sequential(resnet_features, nn.Flatten(), fc)
    return model

def create_model_res50(ch_in: int = 1, ncls: int = None, task: str = "regression"):
    if task == "classification":
        assert ncls is not None
    model = torchvision.models.resnet50(pretrained=True)
    
    if task == "regression":
        fc = nn.Linear(model.fc.in_features, 1)
    else:
        if ncls == 2:
            fc = nn.Linear(model.fc.in_features, 1)
        else:
            fc = nn.Linear(model.fc.in_features, ncls)
    if ch_in != 3:
        resnet_features = nn.Sequential(*list(model.children())[1:-1])
        conv1 = nn.Conv2d(ch_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = nn.Sequential(conv1, resnet_features, nn.Flatten(), fc)
    else:
        resnet_features = nn.Sequential(*list(model.children())[:-1])
        model = nn.Sequential(resnet_features, nn.Flatten(), fc)
    return model

def create_model_res101(ch_in: int = 1, ncls: int = None, task: str = "regression"):
    if task == "classification":
        assert ncls is not None
    model = torchvision.models.resnet101(pretrained=True)
    
    if task == "regression":
        fc = nn.Linear(model.fc.in_features, 1)
    else:
        if ncls == 2:
            fc = nn.Linear(model.fc.in_features, 1)
        else:
            fc = nn.Linear(model.fc.in_features, ncls)
    if ch_in != 3:
        conv1 = nn.Conv2d(ch_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet_features = nn.Sequential(*list(model.children())[1:-1])
        model = nn.Sequential(conv1, resnet_features, nn.Flatten(), fc)
    else:
        resnet_features = nn.Sequential(*list(model.children())[:-1])
        model = nn.Sequential(resnet_features, nn.Flatten(), fc)
    return model

class RegressionResnet(LightningModule):
    def __init__(
        self,
        ch_in: int = 1,
        batch_size: int = 64,
        learning_rate: float = 1e-2,
        which_model: str = "resnet50",
        is_new_model: str = True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ch_in = ch_in
        self.which_model = which_model
        if is_new_model:
            if self.which_model == "resnet50":
                self.model = create_model_res50(ch_in=ch_in).to(self.device)
            elif self.which_model == "resnet101":
                self.model = create_model_res101(ch_in=ch_in).to(self.device)
            elif self.which_model == "resnet18":
                self.model = create_model_res18(ch_in=ch_in).to(self.device)
        else:
            if self.which_model == "resnet50":
                self.model = create_model_res50_old(ch_in=ch_in).to(self.device)
            elif self.which_model == "resnet101":
                self.model = create_model_res101_old(ch_in=ch_in).to(self.device)
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        source1, source2 = batch
        x1, y1 = source1
        x2, y2 = source2
        y_hat_s1 = self(x1.float())
        y_hat_s2 = self(x2.float())
        y_hat = torch.concat([y_hat_s1, y_hat_s2], dim=0)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        source1, source2 = batch
        x1, y1 = source1
        x2, y2 = source2
        y_hat_s1 = self.forward(x1.float())
        y_hat_s2 = self.forward(x2.float())
        y = torch.concat([y1, y2], dim=0)
        y_hat = torch.concat([y_hat_s1, y_hat_s2], dim=0)
        loss = self.loss_fn(y_hat, y)
        # loss = F.l1_loss(y_hat, y, reduction="mean")
        self.log("train_loss", loss)
        return loss
        
    
    def evaluate(self, batch, stage=None):
        source1, source2 = batch
        x1, y1 = source1
        x2, y2 = source2
        y_hat_s1 = self.forward(x1.float())
        y_hat_s2 = self.forward(x2.float())
        y = torch.concat([y1, y2], dim=0)
        y_hat = torch.concat([y_hat_s1, y_hat_s2], dim=0)
        loss = self.loss_fn(y_hat, y)
        # loss = F.l1_loss(y_hat, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="test")
    
    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     momentum = 0.9,
        #     weight_decay=1e-4,
        # )
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
    
class ClassificationResnet(LightningModule):
    def __init__(
        self,
        ch_in: int = 1,
        batch_size: int = 64,
        learning_rate: float = 1e-2,
        ncls: int = None,
        class_weights = None,
        which_model: str = "resnet50"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ch_in = ch_in
        self.ncls = ncls
        self.which_model = which_model
        if self.which_model == "resnet50":
            self.model = create_model_res50(ch_in=ch_in, ncls=ncls, task="classification").to(self.device)
        elif self.which_model == "resnet101":
            self.model = create_model_res101(ch_in=ch_in, ncls=ncls, task="classification").to(self.device)
        elif self.which_model == "resnet18":
            self.model = create_model_res18(ch_in=ch_in, ncls=ncls, task="classification").to(self.device)
        elif self.which_model == "inceptionv3":
            self.model = create_model_inceptionv3(ch_in=ch_in, ncls=ncls, task="classification").to(self.device)
        if ncls == 2:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        source1, source2 = batch
        x1, y1 = source1
        x2, y2 = source2
        y_hat_s1 = self(x1.float())
        y_hat_s2 = self(x2.float())
        if self.ncls == 2:
            y_hat = torch.concat([y_hat_s1, y_hat_s2], dim=0).squeeze()
        else:
            y_hat = torch.concat([y_hat_s1, y_hat_s2], dim=0)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        source1, source2 = batch
        x1, y1 = source1
        x2, y2 = source2
        y_hat_s1 = self.forward(x1.float())
        y_hat_s2 = self.forward(x2.float())
        y = torch.concat([y1, y2], dim=0).float()
        if self.ncls == 2:
            y_hat = torch.concat([y_hat_s1, y_hat_s2], dim=0).squeeze()
        else:
            y_hat = torch.concat([y_hat_s1, y_hat_s2], dim=0)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss
        
    
    def evaluate(self, batch, stage=None):
        source1, source2 = batch
        x1, y1 = source1
        x2, y2 = source2
        y_hat_s1 = self.forward(x1.float())
        y_hat_s2 = self.forward(x2.float())
        y = torch.concat([y1, y2], dim=0).float()
        if self.ncls == 2:
            y_hat = torch.concat([y_hat_s1, y_hat_s2], dim=0).squeeze()
        else:
            y_hat = torch.concat([y_hat_s1, y_hat_s2], dim=0)
        loss = self.loss_fn(y_hat, y)
        if self.ncls == 2:
            class_hat = F.sigmoid(y_hat)
            class_hat = torch.round(class_hat)
        else:
            class_hat = F.softmax(y_hat, dim=1)
            class_hat = torch.argmax(class_hat, dim=1)
        accuracy = torch.sum(class_hat == y) / len(y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_accuracy", accuracy, prog_bar=True)
        return loss, accuracy

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


class MNISTClassificationResnet(LightningModule):
    def __init__(
        self,
        ch_in: int = 1,
        batch_size: int = 64,
        learning_rate: float = 1e-2,
        ncls: int = None,
        class_weights = None,
        which_model: str = "resnet50"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ch_in = ch_in
        self.ncls = ncls
        self.which_model = which_model
        if self.which_model == "resnet50":
            self.model = create_model_res50(ch_in=ch_in, ncls=ncls, task="classification").to(self.device)
        elif self.which_model == "resnet18":
            self.model = create_model_res18(ch_in=ch_in, ncls=ncls, task="classification").to(self.device)
        elif self.which_model == "resnet101":
            self.model = create_model_res101(ch_in=ch_in, ncls=ncls, task="classification").to(self.device)
        elif self.which_model == "vgg19":
            self.model = create_model_vgg19(ch_in=ch_in, ncls=ncls, task="classification").to(self.device)
        elif self.which_model == "inceptionv3":
            self.model = create_model_inceptionv3(ch_in=ch_in, ncls=ncls, task="classification").to(self.device)
        if ncls == 2:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.float())
        if self.ncls == 2:
            loss = self.loss_fn(y_hat.squeeze(), y.float())
            class_hat = F.sigmoid(y_hat.squeeze())
            class_hat = torch.round(class_hat)
        else:            
            loss = self.loss_fn(y_hat, y)
            class_hat = F.softmax(y_hat, dim=1)
            class_hat = torch.argmax(class_hat, dim=1)
        accuracy = torch.sum(class_hat == y) / len(y)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss
        
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self.forward(x.float())
        if self.ncls == 2:
            loss = self.loss_fn(y_hat.squeeze(), y.float())
            class_hat = F.sigmoid(y_hat.squeeze())
            class_hat = torch.round(class_hat)
        else:
            loss = self.loss_fn(y_hat, y)
            class_hat = F.softmax(y_hat, dim=1)
            class_hat = torch.argmax(class_hat, dim=1)
        accuracy = torch.sum(class_hat == y) / len(y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_accuracy", accuracy, prog_bar=True)
        return loss, accuracy

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
    
class MNISTRegressionResnet(LightningModule):
    def __init__(
        self,
        ch_in: int = 1,
        batch_size: int = 64,
        learning_rate: float = 1e-2,
        which_model: str = "resnet50"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ch_in = ch_in
        self.which_model = which_model
        if self.which_model == "resnet50":
            self.model = create_model_res50(ch_in=ch_in, task="regression").to(self.device)
        elif self.which_model == "resnet101":
            self.model = create_model_res101(ch_in=ch_in, task="regression").to(self.device)
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.float())
        if y.dim() == 1:
            loss = self.loss_fn(y_hat.squeeze(), y)
        else:
            loss = self.loss_fn(y_hat, y)
        # loss = F.l1_loss(y_hat, y, reduction="mean")
        self.log("train_loss", loss)
        return loss
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self.forward(x.float())
        if y.dim() == 1:
            loss = self.loss_fn(y_hat.squeeze(), y)
        else:
            loss = self.loss_fn(y_hat, y)
        # loss = F.l1_loss(y_hat, y)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="test")
    
    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     momentum = 0.9,
        #     weight_decay=1e-4,
        # )
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
