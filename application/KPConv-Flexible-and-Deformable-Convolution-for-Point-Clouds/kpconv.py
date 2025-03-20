from mindspore.nn import Cell
from kpconv_utils import *
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore import set_device
from mindspore import Parameter  # Import Parameter

class KPConvRigit(Cell):
    '''
       numKernels : number of kernels
       sigma : sigma parameter of the kernels
       in_channels : number of input features
       out_channels : number of output features
       device :  the device where the model is trainned on
       normalization_factor : initial normalization of the weight matrix (following a normal destribution) # this is a temporary fix
                                                                                                           # will use a init_function 
                                                                                                           # if i make the kernel work
    '''

    def __init__(self, numKernels, sigma, in_channels, out_channels, device = "CPU", normalization_factor = 0.1):
        super(KPConvRigit, self).__init__()
        self.numKernels = numKernels
        self.sigma = sigma
        self.device = device
        # if in_channels == 0, we add an artificial channel containing ones
        if in_channels == 0:
            self.in_channels = 1
        else:
            self.in_channels = in_channels
        self.out_channels = out_channels
        # creating the kernels
        # self.kernels has shape : (numKernels) x 3
        self.kernels = initializeRigitKernels(numKernels)
        # creating per kernel weight matrix
        # self.weights has shape : (numKernels) x (in_channels) x (out_channels)
        self.weights = Parameter(ms.Tensor(ms.numpy.randn(self.numKernels, self.in_channels, self.out_channels), dtype=ms.float32))
        #self.weights*= normalization_factor

    def construct(self, pos, assigned_index, N, M, h=None):
        '''
        - pos : node position
        - assigned_index : connection of nodes
        - N : number of input points
        - M : number of subsampled points
        - h : node feature
        '''
        # Implement message passing manually
        edge_src = assigned_index[0]
        edge_dst = assigned_index[1]
        
        # Validate and correct edge_dst indices
        edge_dst = ops.clip_by_value(edge_dst, 0, M - 1)
        
        pos_i = pos[edge_dst]
        pos_j = pos[edge_src]
        
        if h is None:
            h = ops.Ones()((pos.shape[0], 1), ms.float32)
            
        messages = self.message(h[edge_src], pos_j, pos_i)
        return self.aggregate(messages, edge_dst, M)

    def aggregate(self, messages, edge_dst, M):
        # Call the unsorted_segment_sum function from the ops module
        return ops.unsorted_segment_sum(messages, edge_dst, M)

    def message(self, h_j, pos_j, pos_i):
        '''
        - h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        - pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        - pos_i defines the position of central nodes as shape [num_edges, 3]
        '''
        if h_j is None:
            h_j = ms.Tensor(ms.numpy.ones((pos_j.shape[0], 1)), dtype=ms.float32)
        ## Creating a matrix to save the result after passing it through all the kernels
        added_weights = ms.Tensor(ms.numpy.zeros((pos_j.shape[0], self.in_channels, self.out_channels)), dtype=ms.float32)
        # passing the points through all kernel points to calculate the weight matrix
        for kernel in range(self.numKernels):
            added_weights = added_weights + linearKernel(ops.Sub()(pos_j, pos_i), self.kernels[kernel], self.sigma).unsqueeze(-1).unsqueeze(-1) * self.weights[kernel]
        # unsqueezing to make the tensors have the appropriate size for multiplication
        # the squezzing again to remove signleton dimension
        return ops.Squeeze(1)(ops.BatchMatMul()(h_j.unsqueeze(1), added_weights))

# 主函数
if __name__ == "__main__":
    # Set the device
    set_device("CPU")

    # 初始化模型参数
    numKernels = 10
    sigma = 0.5
    in_channels = 3
    out_channels = 6
    device = "CPU"  # 可以根据实际情况修改为 "cuda:0"

    # 初始化模型
    model = KPConvRigit(numKernels, sigma, in_channels, out_channels, device)

    # 打印模型参数信息
    print("Model parameters:")
    for name, param in model.parameters_and_names():
        print(f"  {name}: {param.shape}")

    # 生成更多随机输入数据
    N = 1000  # 增加输入点的数量
    M = 500   # 增加下采样点的数量
    num_edges = 2000  # 增加边的数量

    pos = ms.Tensor(ms.numpy.randn(N, 3), dtype=ms.float32)  # 节点位置
    assigned_index = ms.Tensor(ms.numpy.randint(0, N, (2, num_edges)), dtype=ms.int32)  # 节点连接
    h = ms.Tensor(ms.numpy.randn(N, in_channels), dtype=ms.float32)  # 节点特征

    # 打印输入数据信息
    print("\nInput data shapes:")
    print(f"  pos: {pos.shape}")
    print(f"  assigned_index: {assigned_index.shape}")
    print(f"  h: {h.shape}")

    # 前向传播
    output = model(pos, assigned_index, N, M, h)
    print("\nOutput shape:", output.shape)

    # 打印输出的部分内容
    print("\nFirst few elements of the output:")
    print(output[:5])
