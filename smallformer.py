import torch
from torch import nn, Tensor
from typing import List
import math



     
        

class LocalMappingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
           
        self.mapping = nn.PReLU(dim)
        self.norm = nn.LayerNorm(dim,elementwise_affine=False)
      
             	   
    def forward(self, x):
    
        x = self.norm(x) 
        x = self.mapping(x[-1])   	
      
        return x
    	

class GlobalMappingUnit(nn.Module):
    
    def __init__(self, dim,heads):
            
        super().__init__()
        self.num_heads = heads
        self.hidden_dim = dim
        self.head_dim = dim // self.num_heads
        self.norm = nn.LayerNorm(dim,elementwise_affine=False)
        assert self.head_dim * self.num_heads == self.hidden_dim 

       
       
    def forward(self, x):
        
        batch_size, seq_len, _ = x.size()

        x = self.norm(x)
        
        P,S = x,x
       

        
        P = P.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        S = S.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        

       
        attention_scores = P @ S.transpose(-1, -2) / math.sqrt(self.head_dim)

       

        
        attention_weights = torch.softmax(attention_scores, dim=-1)

        

       
        context = attention_weights @ S

        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        
        return context         




class SmallFormerBlock(nn.Module):
    def __init__(self, d_model,heads):
        super().__init__()
       
         
        self.local_mapping = LocalMappingUnit(d_model)
        self.global_mapping = GlobalMappingUnit(d_model,heads)
        
    
        
        
        
    def forward(self, x):
                  
        residual = x
        
        x = self.global_mapping(x)
    
        x = x + residual
        
        residual = x
        
        x = self.local_mapping(x)
        
                                          
        out = x + residual
        
        
        return out



class SmallFormer(nn.Module):
    def __init__(self, d_model,heads, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[SmallFormerBlock(d_model,heads) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








