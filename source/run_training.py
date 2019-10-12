import os
import torch
from pytorch_lightning import Trainer
from argparse import ArgumentParser,ArgumentTypeError
from lightning import modules as lm
import warnings
from dataloaders.dataloader_utils import gen_dataloader
warnings.filterwarnings('ignore')



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

    model = lm.LightningBertClass(dl,criterion,args)

    # most basic trainer, uses good defaults
    trainer = Trainer(max_nb_epochs=1, gpus=1, default_save_path='../data/')    
    trainer.fit(model)