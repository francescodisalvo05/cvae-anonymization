
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, AUROC
import pytorch_lightning as pl
import torch


class LinearModule(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 embedding_dimension = 768,
                 lr = 0.001,
                 criterion = torch.nn.CrossEntropyLoss(), 
                 patience = 5,
                 metric_to_monitor = "Val/Loss",
                 metric_to_monitor_mode = "min",
                 weight_decay = 0.0,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dimension = embedding_dimension
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience_scheduler = patience
        self.metric_to_monitor = metric_to_monitor
        self.metric_to_monitor_mode = metric_to_monitor_mode
        self.model = torch.nn.Linear(self.embedding_dimension, self.num_classes)
        self.criterion = criterion
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.auroc = AUROC(task="multiclass", num_classes=self.num_classes)
        self.save_hyperparameters(ignore=['criterion'])

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch[0].to(self.device), batch[1].type(torch.LongTensor).to(self.device)
        return x, y

    def common_step(self, batch, train):
        data, target = batch
        output = self.model(data)
        loss = self.criterion(output, target)
        probas = torch.softmax(output, 1)
        return loss, probas, target

    def training_step(self, batch, batch_idx):
        loss, probas, targets = self.common_step(batch, train=True)
        self.log("Train/Loss", loss, on_epoch=True, on_step=True)
        self.train_probas.append(probas.detach())
        self.train_targets.append(targets.detach())
        return loss

    def on_train_epoch_start(self):
        self.train_probas = []
        self.train_targets = []

    def on_train_epoch_end(self):
        targets, probas = torch.cat(self.train_targets), torch.cat(self.train_probas)
        preds = torch.argmax(probas, 1)
        self.log("Train/Accuracy", self.accuracy(targets, preds))
        self.log("Train/AUROC", self.auroc(probas, targets))
        self.train_probas = []
        self.train_targets = []

    def on_validation_epoch_start(self):
        self.validation_probas = []
        self.validation_targets = []

    def on_validation_epoch_end(self):
        targets, probas = torch.cat(self.validation_targets).int(), torch.cat(self.validation_probas)
        preds = torch.argmax(probas, 1)
        self.log("Val/Accuracy", self.accuracy(targets, preds))
        self.log("Val/AUROC", self.auroc(probas, targets))
        self.validation_probas = []
        self.validation_targets = []

    def validation_step(self, batch, batch_idx):
        loss, probas, targets = self.common_step(batch, train=False)
        self.log("Val/Loss", loss, on_epoch=True, on_step=False)
        self.validation_probas.append(probas.detach())
        self.validation_targets.append(targets.detach())

    def on_test_epoch_start(self):
        self.test_probas = []
        self.test_targets = []
        self.test_max_probas = []
        self.preds = []

    def on_test_epoch_end(self):
        targets, probas = torch.cat(self.test_targets).int(), torch.cat(self.test_probas)
        preds = torch.argmax(probas, 1)
        self.log("Test/AUROC", self.auroc(probas, targets))
        self.preds = preds.detach().cpu()
        self.test_auroc = self.auroc(probas, targets).item()
        
    def test_step(self, batch, batch_idx):
        loss, probas, targets = self.common_step(batch, train=False)
        self.log("Test/Loss", loss, on_epoch=True, on_step=False)
        max_probas, _ = torch.max(probas,1)
        self.test_max_probas.extend(max_probas.cpu().numpy().astype(float))
        self.test_probas.append(probas.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

    def configure_optimizers(self):
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        optimizer = [torch.optim.Adam(params_to_update, lr=self.lr, weight_decay=self.weight_decay)]
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer[0], patience=self.patience_scheduler, mode=self.metric_to_monitor_mode, min_lr=1e-5
            ),
            "monitor": self.metric_to_monitor,
        }
        return optimizer, scheduler