import torch
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from models.bert import BertForWSD
from pytorch_transformers import AdamW
from lightning.metrics_loggers import metrics_logger



class LightningBertClass(pl.LightningModule):

    def __init__(self,_dataloaders,_args,criterion=None):
        super(LightningBertClass, self).__init__()
        """
        pytorch-lightning model class
        _data is a dataloader clasa
        _criterion is the pytorch loss function        
        """
        self.model = BertForWSD(model_type=_args.model_type,token_layer=_args.token_layer)
        self.criterion = criterion
        if not self.criterion:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.combined_dataloaders = _dataloaders
        self.opt_lr = _args.lr
        self.opt_weight_decay = _args.weight_decay
        self.val_metrics = metrics_logger()
        self.test_metrics = metrics_logger()
        self.scheduler = _args.scheduler
        self.hparams = _args


    def predict(self,_model_output):
        logits = F.softmax(_model_output,dim=1)
        y_hat = torch.argmax(logits,dim=1)
        return y_hat

    def forward(self, x):
        b_tokens_tensor, b_sentence_tensor, b_target_token_tensor, y = x
        return self.model.forward(b_tokens_tensor, b_sentence_tensor, b_target_token_tensor)

    def training_step(self, batch, batch_nb):
        # REQUIRED        
        y = batch[3]
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        y = batch[3]
        logits = self.forward(batch)
        y_hat = self.predict(logits)
        self.val_metrics.update(y_hat,y)
        lossval =self.criterion(logits, y)
        return {'val_loss': lossval}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 
                'val_accuracy':self.val_metrics.accuracy,
                #'val_precision':self.val_metrics.precision,
                #'val_recall':self.val_metrics.recall, 
                'val_f1':self.val_metrics.f1,        
                }
        self.val_metrics.reset()
        return {'val_loss': logs['val_loss'],
                #'val_accuracy': logs['val_accuracy'],
                #'val_f1': logs['val_f1'],                
                'log': logs,
                'progress_bar': {'val_loss':avg_loss}}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        y = batch[3]
        logits = self.forward(batch)
        y_hat = self.predict(logits)
        self.test_metrics.update(y_hat,y)
        lossval =self.criterion(logits, y)
        return {'test_loss': lossval}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 
                'test_accuracy':self.test_metrics.accuracy,
                'test_precision':self.test_metrics.precision,
                'test_recall':self.test_metrics.recall, 
                'test_f1':self.test_metrics.f1,        
                }
        self.test_metrics.reset()
        return {'test_loss': logs['test_loss'],
                #'test_accuracy':logs['test_accuracy'],
                #'test_precision':logs['test_precision'],
                #'test_recall':logs['test_recall'],
                #'test_f1':logs['test_f1'],       
                'log': logs,
                'progress_bar': {'test_loss':avg_loss}}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': self.opt_weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}]

        # To reproduce BertAdam specific behavior set correct_bias=False
        optimizer = AdamW(optimizer_grouped_parameters,lr=self.opt_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        if self.scheduler:
            return [optimizer], [scheduler]  
        return optimizer 
        

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        #return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)
        return self.combined_dataloaders.train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        #return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)
        return self.combined_dataloaders.val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return self.combined_dataloaders.test_dataloader