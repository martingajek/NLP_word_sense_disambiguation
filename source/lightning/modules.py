import torch
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
        self.metrics = metrics_logger()


    def predict(self,_model_output):
        logits = F.softmax(_model_output,dim=1)
        y_hat = torch.argmax(logits,dim=1)
        return y_hat

    def forward(self, x):
        b_tokens_tensor, b_sentence_tensor, b_target_token_tensor = x
        return self.model.forward(b_tokens_tensor, b_sentence_tensor, b_target_token_tensor)

    def training_step(self, batch, batch_nb):
        # REQUIRED        
        b_tokens_tensor, b_sentence_tensor, b_target_token_tensor, y = batch
        x = b_tokens_tensor, b_sentence_tensor, b_target_token_tensor
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        b_tokens_tensor, b_sentence_tensor, b_target_token_tensor, y = batch
        x = b_tokens_tensor, b_sentence_tensor, b_target_token_tensor
        logits = self.forward(x)
        y_hat = self.predict(logits)
        self.metrics.update(y_hat,y)
        return {'val_loss': self.criterion(logits, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss, 
                'val_accuracy':self.metrics.accuracy,
                'val_precision':self.metrics.precision,
                'val_recall':self.metrics.recall, 
                'val_f1':self.metrics.f1,        
                }
        self.metrics.reset()
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

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
        return self.combined_dataloaders.subset_val_dataloader