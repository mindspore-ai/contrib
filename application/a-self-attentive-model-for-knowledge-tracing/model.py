import mindspore
from mindspore import nn, ops, Tensor
import numpy as np



class sakt(nn.Cell):
    def __init__(self , ex_total , seq_len, dim , heads, dout ):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim

        self.embd_in = nn.Embedding( 2*ex_total+1, embedding_size  = dim )         # Interaction embedding
        self.embd_ex = nn.Embedding( ex_total+1 , embedding_size = dim )       # Excercise embedding
        self.embd_pos = nn.Embedding( seq_len , embedding_size = dim )

        self.linear = nn.CellList( [nn.Dense(in_channels= dim , out_channels= dim ) for x in range(3)] )   # Linear projection for each embedding
        self.attn = nn.MultiheadAttention(embed_dim= dim , num_heads= heads, dropout= dout )
        self.ffn = nn.CellList([nn.Dense(in_channels= dim , out_channels=d, has_bias= True) for x in range(2)])  # feed forward layers post attention

        self.linear_out = nn.Dense(in_channels= dim , out_channels= 1 , has_bias=True)
        self.layer_norm1 = nn.LayerNorm( [dim] )
        self.layer_norm2 = nn.LayerNorm( [dim] )                           # output with correctnness prediction
        self.drop = nn.Dropout(dout)

    def construct( self , input_in , input_ex):

        ## positional embedding
        pos_in = self.embd_pos( ops.arange(self.seq_len).expand_dims(0) )         #making a tensor of 12 numbers, .unsqueeze(0) for converting to 2d, so as to get a 3d output #print('pos embd' , pos_in.shape)

        ## get the interaction embedding output
        out_in = self.embd_in( input_in )                         # (b, n) --> (b,n,d)
        out_in = out_in + pos_in

        ## split the interaction embeding into v and k ( needs to verify if it is slpited or not)
        value_in = out_in
        key_in   = out_in                                         #print('v,k ', value_in.shape)

        ## get the excercise embedding output
        query_ex = self.embd_ex( input_ex )                       # (b,n) --> (b,n,d) #print(query_ex.shape)

        ## Linearly project all the embedings
        value_in = self.linear[0](value_in).permute(1,0,2)        # (b,n,d) --> (n,b,d)
        key_in = self.linear[1](key_in).permute(1,0,2)
        query_ex =  self.linear[2](query_ex).permute(1,0,2)

        ## pass through multihead attention
        atn_out , _ = self.attn(query_ex , key_in, value_in , attn_mask= Tensor( np.triu(np.ones((self.seq_len ,self.seq_len)), k=1).astype('bool')) )      # lower triangular mask, bool, torch    (n,b,d)
        atn_out = query_ex + atn_out                                  # Residual connection ; added excercise embd as residual because previous ex may have imp info, suggested in paper.
        atn_out = self.layer_norm1( atn_out )                          # Layer norm                        #print('atn',atn_out.shape) #n,b,d = atn_out.shape

        #take batch on first axis
        atn_out = atn_out.permute(1,0,2)                              #  (n,b,d) --> (b,n,d)

        ## FFN 2 layers
        ffn_out = self.drop(self.ffn[1]( nn.ReLU()( self.ffn[0]( atn_out ) )))   # (n,b,d) -->    .view([n*b ,d]) is not needed according to the kaggle implementation
        ffn_out = self.layer_norm2( ffn_out + atn_out )                # Layer norm and Residual connection

        ## sigmoid
        ffn_out = ops.sigmoid(self.linear_out( ffn_out )  )

        return ffn_out


def randomdata():
    input_in = ops.randint( 0 , 49 ,(64 , 12) )
    return input_in, input_in



if __name__ == "__main__":
    ## Testing the model
    E =  50 # total unique excercises
    d = 128 # latent dimension
    n = 12  # sequence length

    d1,d2 = randomdata()

    print( 'Input shape',d1.shape)
    model = sakt( ex_total= E , seq_len= n , dim= d , heads= 8, dout= 0.2 )
    out = model( d1, d2)
    print('Output shape', out.shape)
