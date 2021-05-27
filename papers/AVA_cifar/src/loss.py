import mindspore.nn as nn
import mindspore
from mindspore.ops import operations as P


class LossNet(nn.Cell):
    """modified loss function"""

    def __init__(self, temp=0.1):
        super(LossNet, self).__init__()
        self.concat = P.Concat()
        self.exp = P.Exp()
        self.t = P.Transpose()
        self.diag_part = P.DiagPart()
        self.matmul = P.MatMul()
        self.sum = P.ReduceSum()
        self.sum_keep_dim = P.ReduceSum(keep_dims=True)
        self.log = P.Log()
        self.mean = P.ReduceMean()
        self.shape = P.Shape()
        self.eye = P.Eye()
        self.temp = temp

    def diag_part_new(self, input, batch_size):
        eye_matrix = self.eye(batch_size, batch_size, mindspore.float32)
        input = input * eye_matrix
        input = self.sum_keep_dim(input, 1)
        return input

    def construct(self, z_aux, y, x, label):
        batch_size = self.shape(x)[0]
        embed_size = self.shape(x)[1]

        perm = (1, 0)
        mat_x_x = self.exp(self.matmul(x, self.t(x, perm) / self.temp))
        mat_y_y = self.exp(self.matmul(y, self.t(y, perm) / self.temp))
        mat_x_y = self.exp(self.matmul(x, self.t(y, perm) / self.temp))

        mat_aux_x = self.exp(self.matmul(x, self.t(z_aux, perm) / self.temp))
        mat_aux_y = self.exp(self.matmul(y, self.t(z_aux, perm) / self.temp))
        mat_aux_z_x = self.exp(self.matmul(z_aux, self.t(x, perm) / self.temp))
        mat_aux_z_y = self.exp(self.matmul(z_aux, self.t(y, perm) / self.temp))

        loss_mutual = self.mean(-2 * self.log(self.diag_part_new(mat_x_y, batch_size) / (
                1e-10 + self.sum_keep_dim(mat_x_y, 1) - self.diag_part_new(mat_x_y, batch_size) + self.sum_keep_dim(
            mat_x_x, 1) - self.diag_part_new(mat_x_x, batch_size) + self.sum_keep_dim(mat_y_y, 1) - self.diag_part_new(
            mat_y_y, batch_size))))

        loss_aux_x = self.mean(-self.log((self.diag_part_new(mat_aux_x, batch_size) / (
                self.sum_keep_dim(mat_aux_x, 1) - self.diag_part_new(mat_aux_x, batch_size)))))

        loss_aux_y = self.mean(-self.log((self.diag_part_new(mat_aux_y, batch_size) / (
                    self.sum_keep_dim(mat_aux_y, 1) - self.diag_part_new(mat_aux_y, batch_size)))))

        loss_aux_z_x = self.mean(-self.log((self.diag_part_new(mat_aux_z_x, batch_size) / (
                    self.sum_keep_dim(mat_aux_z_x, 1) - self.diag_part_new(mat_aux_z_x, batch_size)))))

        loss_aux_z_y = self.mean(-self.log((self.diag_part_new(mat_aux_z_y, batch_size) / (
                    self.sum_keep_dim(mat_aux_z_y, 1) - self.diag_part_new(mat_aux_z_y, batch_size)))))

        loss = loss_mutual + loss_aux_x + loss_aux_y + loss_aux_z_x + loss_aux_z_y

        return loss
