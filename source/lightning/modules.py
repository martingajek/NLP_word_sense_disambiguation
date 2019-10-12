import torch
import pytorch_lightning as pl
from models.bert import BertForWSD
from pytorch_transformers import AdamW



class LightningBertClass(pl.LightningModule):

    def __init__(self,_dataloaders,_criterion,_args):
        super(LightningBertClass, self).__init__()
        """
        pytorch-lighnting model class
        _data is a dataloader clasa
        _criterion is the pytorch loss function        
        """
        self.model = BertForWSD(model_type=_args.model_type,token_layer=_args.token_layer)
        self.combined_dataloaders = _dataloaders
        self.criterion = _criterion
        self.opt_lr = _args.lr
        self.opt_weight_decay = _args.weight_decay

    def forward(self, x):
        #x = (tens.to(self.device,non_blocking=self.non_blocking) for tens in x)
        b_tokens_tensor, b_sentence_tensor, b_target_token_tensor = x
        return self.model.forward(b_tokens_tensor, b_sentence_tensor, b_target_token_tensor)

    def training_step(self, batch, batch_nb):
        # REQUIRED        
        #batch = (tens.to(self.device,non_blocking=self.non_blocking) for tens in batch)
        b_tokens_tensor, b_sentence_tensor, b_target_token_tensor, y = batch
        x = b_tokens_tensor, b_sentence_tensor, b_target_token_tensor
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        b_tokens_tensor, b_sentence_tensor, b_target_token_tensor, y = batch
        x = b_tokens_tensor, b_sentence_tensor, b_target_token_tensor
        y_hat = self.forward(x)
        return {'val_loss': self.criterion(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
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