from torch.autograd import Function
from torch import nn
from torch import arange, zeros_like


class TokenClsFunction(Function):
    """
    Torch Function that takes as input the hidden state from BERT with shape
    [Size_batch, Max_sentence_size, 758] and a 1D tensor of indexes to highlighted
    tokens and returns [Size_batch, 758] tensor which corresponds to the hidden state
    of each highlighted token in each instance of the batch. See
    https://arxiv.org/pdf/1908.07245.pdf
    """
    
        
    @staticmethod
    def forward(ctx, input, target_token_tensor):
        ctx.save_for_backward(input,target_token_tensor)
        target_token_tensor.requires_grad = False
        flattened_target_tensor = target_token_tensor.flatten()
        return input[arange(flattened_target_tensor.shape[0]),flattened_target_tensor,:]

        
    @staticmethod
    def backward(ctx, grad_output):
        input1,target_token_tensor = ctx.saved_tensors
        grad = zeros_like(input1)
        flattened_target_tensor = target_token_tensor.flatten()
        # gradient only flows to specific indexes of target tensor
        grad[arange(flattened_target_tensor.shape[0]),flattened_target_tensor,:] = grad_output
        return grad, zeros_like(target_token_tensor)  
    
class SentClsFunction(Function):
    """
    Torch Layer that takes as input the hidden state from BERT with shape
    [Size_batch, Max_sentence_size, 758] and returns tensor which corresponds 
    to the hidden state of the CLS (first) token in each instance of the batch. See 
    https://arxiv.org/pdf/1908.07245.pdf
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input[:,0,:]
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = zeros_like(input)
        # gradient only flows only to indexes 0 of target tensor
        # index corresponding to [CLS] token
        grad[:,0,:] = grad_output
        return grad
    
class SentClsWsFunction(SentClsFunction):
    # Copy of SentCLsLayer class
    pass


class TokenClsLayer(nn.Module):
    def __init__(self):
        """
        Pytorch Layer implementing the TokenClsFunction
        """
        super(TokenClsLayer, self).__init__()
        self.tcf = TokenClsFunction.apply
        
    def forward(self, features, token_indexes):
        """
        Passing 2 tensors, the first one is the output or the last hidden state from the transformer model
        the 2nd one token_indexes corresponds to the indexes of highlighted tokens in each batch
        """
        
        return self.tcf(features,token_indexes)
    

class SentClsLayer(nn.Module):
    def __init__(self):
        """
        Pytorch Layer implementing the SentClsLayer
        """
        super(SentClsLayer, self).__init__()
        self.scf = SentClsFunction.apply
        
    def forward(self, features):
        """
        Passing 1 tensors, namely the output or the last hidden state from the transformer model
        """
        
        return self.scf(features)