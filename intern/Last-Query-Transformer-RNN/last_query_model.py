import mindspore
from mindspore import nn, ops
import numpy as np
import copy
mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE, device_target="CPU")
class Feed_Forward_block(nn.Cell):
    """
    out = Relu(M_out*w1 + b1) * w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Dense(in_channels=dim_ff, out_channels=dim_ff)
        self.layer2 = nn.Dense(in_channels=dim_ff, out_channels=dim_ff)

    def construct(self, ffn_in):
        return self.layer2(ops.relu(self.layer1(ffn_in)))

class LastQueryModel(nn.Cell):
    """
    Embedding --> MLH --> LSTM
    """
    def __init__(self, dim_model, heads_en, total_ex, total_cat, total_in, seq_len, use_lstm=True):
        super().__init__()
        self.seq_len = seq_len
        self.embd_ex = nn.Embedding(total_ex, embedding_size=dim_model)
        self.embd_cat = nn.Embedding(total_cat, embedding_size=dim_model)
        self.embd_in = nn.Embedding(total_in, embedding_size=dim_model)

        self.multi_en = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_en,dropout=0.1)
        self.ffn_en = Feed_Forward_block(dim_model)
        self.layer_norm1 = nn.LayerNorm((dim_model,))
        self.layer_norm2 = nn.LayerNorm((dim_model,))

        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=dim_model, hidden_size=dim_model, num_layers=1)

        self.out = nn.Dense(in_channels=dim_model, out_channels=1)

    def construct(self, in_ex, in_cat, in_in, first_block=True):
        first_block = True
        if first_block:
            in_ex = self.embd_ex( in_ex )
            in_ex = nn.Dropout(0.1)(in_ex)

            in_cat = self.embd_cat( in_cat )
            in_cat = nn.Dropout(0.1)(in_cat)

            #print("response embedding ", in_in.shape , '\n' , in_in[0])
            in_in = self.embd_in(in_in)
            in_in = nn.Dropout(0.1)(in_in)

            #in_pos = self.embd_pos( in_pos )
            #combining the embedings
            out = in_ex + in_cat + in_in #+ in_pos                      # (b,n,d)

        else:
            out = in_ex
        
        #in_pos = get_pos(self.seq_len)
        #in_pos = self.embd_pos( in_pos )
        #out = out + in_pos                                      # Applying positional embedding

        out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape )
        
        #Multihead attention                            
        n,_,_ = out.shape
        out = self.layer_norm1( out )                           # Layer norm
        skip_out = out 
        print('MLH out shape', out.shape)
        out, attn_wt = self.multi_en( out[-1:,:,:] , out , out )         # Q,K,V
        #                        #attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        #print('MLH out shape', out.shape)
        out = out + skip_out                                    # skip connection

        #LSTM
        if self.use_lstm:
            out,_ = self.lstm( out )                                  # seq_len, batch, input_size
            out = out[-1:,:,:]

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d)
        out = self.layer_norm2( out )                           # Layer norm 
        skip_out = out
        out = self.ffn_en( out )
        out = out + skip_out                                    # skip connection

        out = self.out( out )

        return out.squeeze(-1), 0

def get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])

def get_mask(seq_len):
    return mindspore.from_numpy( np.triu(np.ones((1 ,seq_len)), k=1).astype('bool'))

def random_data(bs, seq_len, total_ex, total_cat, total_in=2):
    ex = ops.randint( 0 , total_ex ,(bs , seq_len) )
    cat = ops.randint( 0 , total_cat ,(bs , seq_len) )
    res = ops.randint( 0 , total_in ,(bs , seq_len) )
    return ex,cat, res

seq_len = 100
total_ex = 1200
total_cat = 234
total_in = 2

in_ex, in_cat, in_in = random_data(64, seq_len, total_ex, total_cat, total_in)

model = LastQueryModel(dim_model=128,
            heads_en=1,
            total_ex=total_ex,
            total_cat=total_cat,
            seq_len=seq_len,
            total_in=2
            )

outs = model(in_ex, in_cat, in_in)

print('Output LSTM shape - ', outs[0].shape)
