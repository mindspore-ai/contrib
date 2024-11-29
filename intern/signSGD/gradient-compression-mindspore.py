import mindspore as ms
from mindspore import ops,Tensor
import numpy as np

def quantize(x,input_compress_settings={}):
    compress_settings={'n':6}
    compress_settings.update(input_compress_settings)
    #assume that x is a mindspore tensor
    
    n=compress_settings['n']
    #print('n:{}'.format(n))
    x=x.float()
    x_norm=Tensor.norm(x,float('inf'))
    #print(x_norm)
    sgn_x=ops.sign(x)
    #print(f"quantize:sgn_x={sgn_x}")
    p=ops.div(ops.abs(x),x_norm)
    #print(p)
    renormalize_p=ops.mul(p,n)
    #print(renormalize_p)
    floor_p=ops.floor(renormalize_p)
    #print(floor_p)
    compare=ops.rand_like(floor_p)
    #print(compare)
    final_p=renormalize_p-floor_p
    #print(final_p)
    margin=(compare < final_p).float()
    #print(margin)
    xi=(floor_p+margin)/n
    #print(xi)
    
    
    Tilde_x=x_norm*sgn_x*xi
    return Tilde_x

def sparse_randomized(x,input_compress_settings={}):
    max_iteration=10000
    compress_settings={'p':0.8}
    compress_settings.update(input_compress_settings)
    #p=compress_settings['p']
    #vec_x=x.flatten()
    #out=torch.dropout(vec_x,1-p,train=True)
    #out=out/p
    vec_x=x.flatten()
    #print(vec_x)
    d = int(len(vec_x))
    #print(d)
    p=compress_settings['p']
    
    abs_x=ops.abs(vec_x)
    #print(abs_x)
    #d=ops.prod(Tensor(x.size()))
#     print(f"1:{p*d*abs_x/ops.sum(abs_x)}")
#     print(f"2:{ops.ones_like(abs_x)}")
    x1=p*d*abs_x/ops.sum(abs_x)
    x2=ops.ones_like(abs_x)
    x1=x1.asnumpy()
    x2=x2.asnumpy()
    x1=x1.tolist()
    x2=x2.tolist()
    x3=[]
    for i in range(0,len(x1)):
        x3.append(min(x1[i],x2[i]))
    out=Tensor(x3)
    #print(out)
    i=0
    while True:
        i+=1
        #print(i)
        if i>=max_iteration:
            raise ValueError('Too much operations!')
        temp=ops.stop_gradient(out)
            
        cI=1-ops.equal(out,1).float()
        c=(p*d-d+ops.sum(cI))/ops.sum(out*cI)
        if c<=1:
            break
        x1=c*out
        x2=ops.ones_like(out)
        x1=x1.asnumpy()
        x2=x2.asnumpy()
        x1=x1.tolist()
        x2=x2.tolist()
        x3=[]
        for i in range(0,len(x1)):
            x3.append(min(x1[i],x2[i]))
        out=Tensor(x3)
#         out=ops.min(c*out,ops.ones_like(out))
        if ops.sum(1-ops.equal(out,temp)):
            break
    
    z=ops.rand_like(out)
    #print(z)
    out=vec_x*(z<out).float()/out
    #print(out)
    out=out.reshape(x.shape)

    return out

def one_bit(x,input_compress_settings={}):
    
    x_norm=Tensor.norm(x,float('inf'))
    #print(x_norm)
    sgn_x=ops.sign(x)
    #print(sgn_x)
    compressed_x=x_norm*sgn_x
    
    return compressed_x

def sparse_top_k(x,input_compress_settings={}):
    compress_settings={'k':1/32}
    compress_settings.update(input_compress_settings)
    k=compress_settings['k']
    vec_x=x.flatten()
    d = int(len(vec_x))
    #print(d)
    k =int(np.ceil(d*k))
    #print(k)
    indices = ops.abs(vec_x).topk(k)[1]
    out_x = ops.zeros_like(vec_x)
    out_x[indices] = vec_x[indices]
    out_x=out_x.reshape(x.shape)
    #print(x.shape)
    return out_x

if __name__ == '__main__':
    inputx=Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), ms.float32)
    print(inputx)
    output1=quantize(inputx)#有问题
    print(output1)
    output2=sparse_randomized(inputx)#有问题
    print(output2)
    output3=one_bit(inputx)#有问题
    print(output3)
    output4=sparse_top_k(inputx)
    print(output4)