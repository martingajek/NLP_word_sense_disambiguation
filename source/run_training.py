
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import torch
from pytorch_lightning import Trainer
from argparse import ArgumentParser,ArgumentTypeError
from lightning import modules as lm
from lightning.utils import str2bool
from dataloaders.dataloader_utils import gen_dataloader
from multiprocessing import cpu_count



if __name__ == '__main__':
    
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
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='Fraction of epoch at which to check validation set')
    parser.add_argument("--default_save_path", type=str, default="../data/",
                        help="Save path for logs and checkpoints log directory for Tensorboard log output")
    parser.add_argument("--model", type=str, default='bert') # can be xlnet as well
    parser.add_argument("--model_type", type=str, default='bert-base-uncased',
                        help="bert model: default is bert-base-uncased")
    parser.add_argument("--token_layer", type=str, default='token-cls',
                        help="bert token layer type: default is token-cls")
    parser.add_argument("--weak_supervision", type=str2bool, default=False,
                        help="Enable context gloss weak supervision")
    #parser.add_argument("--num_workers", type=int, default=0,
    #                    help="Enable non_blocking argument in pytorch to speedup GPU memory transfers")
    parser.add_argument("--preprocess_inputs", type=str2bool, default=False,
                        help="Preprocess input data (Tokenize and generate input embeddings)")
    parser.add_argument("--input_len", type=int, default=128,
                        help="Sentence max length/padding size")
    parser.add_argument("--comments", type=str, default='',
                        help="Comments to go into tensorboard logs")
    parser.add_argument('--class_weights', nargs='+', type=float,default=[1.0,1.0], 
                        help="class weights to be used in the loss function")
    args = parser.parse_args()

    print('Running with {}'.format(args))
    print()
    # ## Process Data
    if args.preprocess_inputs: print('Preprocessing data')
    else:  print('Loading data...')     
    dl = gen_dataloader(args.data_path,args.test_data_path,args.batch_size,
                        preprocess_inputs=args.preprocess_inputs,
                        sample_size=None,
                        weak_supervision=args.weak_supervision,
                        val_sample_dataloader=True,
                        pin_memory=True,
                        num_workers=cpu_count()*2,
                        tokenizer_type = args.model_type,
                        input_len = args.input_len            
                        )
    print()

    #weight = torch.FloatTensor(class_weights).to(device)
    model = lm.LightningBertClass(dl,args)

    # most basic trainer, uses good defaults
    trainer = Trainer(max_nb_epochs=args.epochs, 
                    gpus=torch.cuda.device_count(), 
                    default_save_path=args.default_save_path,
                    val_check_interval=args.val_check_interval,
                    distributed_backend='ddp',)    
    trainer.fit(model)