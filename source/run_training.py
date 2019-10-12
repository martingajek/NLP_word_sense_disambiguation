import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import pytorch_lightning as pl
from models.bert import BertForWSD
from pytorch_transformers import AdamW
from dataloaders.dataloader_utils import gen_dataloader
from pytorch_lightning import Trainer
from argparse import ArgumentParser,ArgumentTypeError



class LightningBertClass(pl.LightningModule):

    def __init__(self):
        super(LightningBertClass, self).__init__(_dataloaders,_criterion,_args)
        """
        pytorch-lighnting model class
        _data is a dataloader clasa
        _criterion is the pytorch loss function        
        """
        self.model = BertForWSD(model_type=args.model_type,token_layer=args.token_layer)
        self.combined_dataloaders = _dataloaders
        self.criterion = _criterion
        self.opt_lr = _args.lr
        self.opt_weight_decay = _args.weight_decay

    def forward(self, x):
        x = (tens.to(self.device,non_blocking=self.non_blocking) for tens in x)
        b_tokens_tensor, b_sentence_tensor, b_target_token_tensor = x
        return self.bert.forward(b_tokens_tensor, b_sentence_tensor, b_target_token_tensor)

    def training_step(self, batch, batch_nb):
        # REQUIRED        
        batch = (tens.to(self.device,non_blocking=self.non_blocking) for tens in batch)
        b_tokens_tensor, b_sentence_tensor, b_target_token_tensor, y = batch
        x = b_tokens_tensor, b_sentence_tensor, b_target_token_tensor
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
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


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')



    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/preprocessed/fullcorpus.feather",
                        help="Input data path")
    parser.add_argument("--test_data_path", type=str, default="",
                        help="Input test data path")
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 64)')
    #parser.add_argument('--val_batch_size', type=int, default=1000,
    #                    help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='SGD momentum (default: 0.5)')
    #parser.add_argument('--momentum', type=float, default=0.5,
    #                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="../data/logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--checkpoint_dir", type=str, default="../data/model_checkpoints/",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--model", type=str, default='bert') # can be xlnet as well
    parser.add_argument("--model_type", type=str, default='bert-base-uncased',
                        help="bert model: default is bert-base-uncased")
    parser.add_argument("--token_layer", type=str, default='token-cls',
                        help="bert token layer type: default is token-cls")
    parser.add_argument("--weak_supervision", type=str2bool, default=False,
                        help="Enable context gloss weak supervision")
    parser.add_argument("--optimize_gpu_mem", type=str2bool, default=False,
                        help="Enable non_blocking argument in pytorch to speedup GPU memory transfers")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Enable non_blocking argument in pytorch to speedup GPU memory transfers")
    parser.add_argument("--preprocess_inputs", type=str2bool, default=False,
                        help="Preprocess input data (Tokenize and generate input embeddings)")
    parser.add_argument("--input_len", type=int, default=128,
                        help="Sentence max length/padding size")
    parser.add_argument("--comments", type=str, default='',
                        help="COmments to go into tensorboard logs")
    parser.add_argument('--class_weights', nargs='+', type=float,default=[1.0,1.0], 
                        help="class weights to be used in the loss function")
    args = parser.parse_args()

    print('Running with {}'.format(args))
    print()
    # ## Process Data
    print('Preprocessing data')      
    dl = gen_dataloader(args.data_path,args.test_data_path,args.batch_size,
                        preprocess_inputs=args.preprocess_inputs,
                        sample_size=None,
                        weak_supervision=args.weak_supervision,
                        val_sample_dataloader=True,
                        pin_memory=args.optimize_gpu_mem,
                        num_workers=args.num_workers,
                        tokenizer_type = args.model_type,
                        input_len = args.input_len            
                        )
    print()

    #weight = torch.FloatTensor(class_weights).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    model = LightningBertClass(dl,criterion,args)

    # most basic trainer, uses good defaults
    trainer = Trainer()    
    trainer.fit(model)   