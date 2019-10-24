
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)

import os
import torch
from pytorch_lightning import Trainer
from argparse import ArgumentParser,ArgumentTypeError
from test_tube import HyperOptArgumentParser
from lightning import modules as lm
from lightning.utils import str2bool
from dataloaders.dataloader_utils import gen_dataloader
from multiprocessing import cpu_count
from test_tube import Experiment


def main(hparams,*args,**kwargs):

    dl = gen_dataloader(hparams.data_path,hparams.test_data_path,hparams.batch_size,
                        weak_supervision=hparams.weak_supervision,
                        val_dataloader=True,
                        pin_memory=False,
                        train_samples=hparams.train_samples,
                        val_sample_size=hparams.val_sample_size,
                        num_workers=cpu_count()*2,
                        tokenizer_type = hparams.model_type,
                        input_len = hparams.input_len,
                        )

    
    #weight = torch.FloatTensor(class_weights).to(device)
    model = lm.LightningBertClass(dl,hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer(
                    fast_dev_run=False,
                    checkpoint_callback=hparams.enable_checkpoints,
                    max_nb_epochs=hparams.epochs, 
                    gpus=torch.cuda.device_count(), 
                    default_save_path=hparams.default_save_path,
                    val_check_interval=hparams.val_check_interval,
                    distributed_backend=hparams.distributed_backend,)    
    trainer.fit(model)


if __name__ == '__main__':
    
    parser = HyperOptArgumentParser(strategy='random_search')
    parser.add_argument("--data_path", type=str, default="../data/preprocessed/fullcorpus.feather",
                        help="Input data path")
    parser.add_argument("--test_data_path", type=str, default="",
                        help="Input test data path")
    parser.add_argument("--enable_checkpoints", type=bool, default=True,
                        help="Save chackpoints to path")                    
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='Fraction of epoch at which to check validation set')
    parser.add_argument("--default_save_path", type=str, default="../data/",
                        help="Save path for logs and checkpoints log directory for Tensorboard log output")
    parser.add_argument('--epochs', type=int, default=10,
                       help='number of epochs to train (default: 10)')
    parser.add_argument("--model", type=str, default='bert') # can be xlnet as well
    parser.add_argument("--model_type", type=str, default='bert-base-uncased',choices=['bert-base-uncased','bert-base-cased'],
                        help="bert model: default is bert-base-uncased")
    parser.add_argument("--input_len", type=int, default=128,
                        help="Sentence max length/padding size",choices=[128,256])
    parser.add_argument('--class_weights', nargs='+', type=float,default=(1.0,1.0), 
                        help="class weights to be used in the loss function")
    parser.add_argument('--train_samples', type=int, default=None,
                        help="Number trials for hyperparameter optimization")                    
    parser.add_argument('--val_sample_size', type=float, default=0.1,
                        help="fraction of test dataset to use for validation")                    
    parser.add_argument('--distributed_backend', type=str, default=None,choices=[None,'dp','dpp','dpp2'],
                        help="distirbuted backend type")                    
    parser.opt_list('--scheduler', type=bool,default=False,options=[True,False],tunable=True,
                        help="Enable cosine scheduler for optimizer")
    parser.opt_list('--batch_size', type=int, default=16, options=[8,16,32,64],
                        help='input batch size for training (default: 64)',tunable=True)    
    parser.opt_range('--lr', type=float, default=2e-5,low=2E-8, high=2E-3,nb_samples=10, log_base=10,
                        help='learning rate (default: 0.01)',tunable=True)
    parser.opt_range('--weight_decay', type=float, default=0.01,low=1E-5,
                        high=0.2,nb_samples=10, log_base=10,
                        help='SGD momentum (default: 0.5)',tunable=True)
    parser.opt_list("--token_layer", type=str, default='token-cls',
                        help="bert token layer type: default is token-cls",
                        options=['token-cls','sent-cls','sent-cls-ws'],tunable=False)
    parser.opt_list("--weak_supervision", type=bool, default=True,options=[True,False],
                    help="Enable context gloss weak supervision",tunable=False)
    
    hyperparams = parser.parse_args()

     main(hyperparams)
 