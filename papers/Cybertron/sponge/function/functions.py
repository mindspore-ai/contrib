# Copyright 2021-2022 The AIMM Group at Shenzhen Bay Laboratory & Peking University
#
# Developer: Yi Isaac Yang, Diqing Chen, Jun Zhang, Yijie Xia, Yupeng Huang
#
# Contact: yangyi@szbl.ac.cn
#
# This code is a part of MindSponge.
#
# The Cybertron-Code is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Common functions
"""

import numpy as np
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import ops
from mindspore import nn
from mindspore import ms_function
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

PI = 3.141592653589793238462643383279502884197169399375105820974944592307

inv = ops.Inv()
keepdim_sum = ops.ReduceSum(keep_dims=True)
keepdim_mean = ops.ReduceMean(keep_dims=True)
keepdim_prod = ops.ReduceProd(keep_dims=True)
keep_norm_last_dim = nn.Norm(axis=-1, keep_dims=True)
norm_last_dim = nn.Norm(axis=-1, keep_dims=False)
reduce_any = ops.ReduceAny()
reduce_all = ops.ReduceAll()
concat_last_dim = ops.Concat(-1)
concat_penulti = ops.Concat(-2)


@ms_function
def pbc_box_reshape(pbc_box: Tensor, ndim: int) -> Tensor:
    r"""Reshape the pbc_box as the same ndim.

    Args:
        pbc_box (Tensor):   Tensor of shape (B,D). Data type is float.
        ndim (int):         The rank (ndim) of the pbc_box

    Returns:
        pbc_box (Tensor):   Tensor of shape (B,1,..,1,D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    if ndim <= 2:
        return pbc_box
    shape = pbc_box.shape[:1] + (1,) * (ndim - 2) + pbc_box.shape[-1:]
    return F.reshape(pbc_box, shape)


@ms_function
def periodic_image(position: Tensor, pbc_box: Tensor, shift: float = 0) -> Tensor:
    r"""calculate the periodic image of the PBC box

    Args:
        position (Tensor):  Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.
        shift (float):      Shift of PBC box. Default: 0

    Returns:
        image (Tensor): Tensor of shape (B, ..., D). Data type is int32.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    pbc_box = pbc_box_reshape(F.stop_gradient(pbc_box), position.ndim)
    image = -F.floor(position / pbc_box - shift)
    return F.cast(image, ms.int32)


@ms_function
def displace_in_box(position: Tensor, pbc_box: Tensor, shift: float = 0) -> Tensor:
    r"""displace the positions of system in a PBC box

    Args:
        position (Tensor):  Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.
        shift (float):      Shift of PBC box. Default: 0

    Returns:
        position_in box (Tensor):   Tensor of shape (B, ..., D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    pbc_box = pbc_box_reshape(F.stop_gradient(pbc_box), position.ndim)
    image = -F.floor(position / pbc_box - shift)
    return position + pbc_box * image


@ms_function
def vector_in_box(vector: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Make the vector at the range from -0.5 box to 0.5 box
        at perodic bundary condition. (-0.5box < difference < 0.5box)

    Args:
        vector (Tensor):        Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        diff_in_box (Tensor):   Tensor of shape (B, ..., D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    pbc_box = pbc_box_reshape(pbc_box, vector.ndim)
    box_nograd = F.stop_gradient(pbc_box)
    inv_box = msnp.reciprocal(box_nograd)
    vector -= box_nograd * F.floor(vector * inv_box + 0.5)
    return  vector * inv_box * pbc_box

@ms_function
def get_vector_without_pbc(initial: Tensor, terminal: Tensor, _pbc_box=None) -> Tensor:
    r"""Compute vector from initial point to terminal point without perodic bundary condition.

    Args:
        initial (Tensor):   Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of initial point
        terminal (Tensor):  Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of terminal point
        _pbc_box (None):    Dummy.

    Returns:
        vector (Tensor):    Tensor of shape (B, ..., D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """
    #pylint: disable=invalid-name

    return terminal - initial


@ms_function
def get_vector_with_pbc(initial: Tensor, terminal: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Compute vector from initial point to terminal point at perodic bundary condition.

    Args:
        initial (Tensor):   Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of initial point
        terminal (Tensor):  Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of terminal point
        pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.

    Returns:
        vector (Tensor):    Tensor of shape (B, ..., D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    return vector_in_box(terminal-initial, pbc_box)

@ms_function
def get_vector(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None) -> Tensor:
    r"""Compute vector from initial point to terminal point.

    Args:
        initial (Tensor):   Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of initial point
        terminal (Tensor):  Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of terminal point
        pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.
                            Default: None

    Returns:
        vector (Tensor):    Tensor of shape (B, ..., D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    vector = terminal - initial
    if pbc_box is None:
        return vector
    return vector_in_box(vector, pbc_box)


@ms_function
def gather_vectors(tensor: Tensor, index: Tensor) -> Tensor:
    r"""Gather vectors from the penultimate axis (axis=-2) of the tensor according to index.

    Args:
        tensor (Tensor):    Tensor of shape (B, A, D).
        index (Tensor):     Tensor of shape (B, ...,). Data type is int.

    Returns:
        vector (Tensor):    Tensor of shape (B, ..., D).

    """

    if index.shape[0] == 1:
        return F.gather(tensor, index[0], -2)
    if tensor.shape[0] == 1:
        return F.gather(tensor[0], index, -2)

    # (B, N, M)
    shape0 = index.shape
    # (B, N*M, 1) <- (B, N, M)
    index = F.reshape(index, (shape0[0], -1, 1))
    # (B, N*M, D) <- (B, N, D)
    neigh_atoms = msnp.take_along_axis(tensor, index, axis=-2)
    # (B, N, M, D) <- (B, N, M) + (D,)
    output_shape = shape0 + tensor.shape[-1:]

    # (B, N, M, D)
    return F.reshape(neigh_atoms, output_shape)


@ms_function
def gather_values(tensor: Tensor, index: Tensor) -> Tensor:
    r"""Gather values from the last axis (axis=-1) of the tensor according to index.

    Args:
        tensor (Tensor):    Tensor of shape (B, X).
        index (Tensor):     Tensor of shape (B, ...,). Data type is int.

    Returns:
        value (Tensor): Tensor of shape (B, ...,).

    """

    if index.shape[0] == 1:
        return F.gather(tensor, index[0], -1)
    if tensor.shape[0] == 1:
        return F.gather(tensor[0], index, -1)

    # (B, N, M)
    origin_shape = index.shape
    # (B, N*M) <- (B, N, M)
    index = F.reshape(index, (origin_shape[0], -1))

    # (B, N*M)
    neigh_values = F.gather_d(tensor, -1, index)

    # (B, N, M)
    return F.reshape(neigh_values, origin_shape)


@ms_function
def calc_distance_without_pbc(position_a: Tensor, position_b: Tensor, _pbc_box=None) -> Tensor:
    r"""Compute distance between position A and B without perodic bundary condition.

    Args:
        position_a (Tensor):    Tensor of shape (..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (..., D). Data type is float.
        _pbc_box (None):        Dummy.

    Returns:
        distance (Tensor):  Tensor of shape (..., 1). Data type is float.

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    """
    #pylint: disable=invalid-name

    vec = get_vector_without_pbc(position_a, position_b)
    return keep_norm_last_dim(vec)


@ms_function
def calc_distance_with_pbc(position_a: Tensor, position_b: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Compute distance between position A and B at perodic bundary condition.

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        distance (Tensor):  Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    vec = get_vector_with_pbc(position_a, position_b, pbc_box)
    return keep_norm_last_dim(vec)


@ms_function
def calc_distance(position_a: Tensor, position_b: Tensor, pbc_box: Tensor = None) -> Tensor:
    r"""Compute distance between position A and B

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        distance (Tensor):  Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    vec = get_vector_without_pbc(position_a, position_b)
    if pbc_box is not None:
        vec = vector_in_box(vec, pbc_box)
    return keep_norm_last_dim(vec)


@ms_function
def calc_angle_between_vectors(vector1: Tensor, vector2: Tensor) -> Tensor:
    r"""Compute angle between two vectors.

    Args:
        vector1 (Tensor):    Tensor of shape (..., D). Data type is float.
        vector1 (Tensor):    Tensor of shape (..., D). Data type is float.

    Returns:
        angle (Tensor):  Tensor of shape (..., 1). Data type is float.

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    """

    # [..., 1] <- [..., D]
    dis1 = keep_norm_last_dim(vector1)
    dis2 = keep_norm_last_dim(vector2)
    dot12 = keepdim_sum(vector1 * vector2, -1)
    # [..., 1]
    cos_theta = dot12 / dis1 / dis2
    return F.acos(cos_theta)


@ms_function
def calc_angle_without_pbc(position_a: Tensor, position_b: Tensor, position_c: Tensor) -> Tensor:
    r"""Compute angle formed by three positions A-B-C without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape (..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (..., D). Data type is float.
        position_c (Tensor):    Tensor of shape (..., D). Data type is float.

    Returns:
        angle (Tensor):  Tensor of shape (..., 1). Data type is float.

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    """

    # (...,D)
    vec_ba = get_vector_without_pbc(position_b, position_a)
    vec_bc = get_vector_without_pbc(position_b, position_c)
    return calc_angle_between_vectors(vec_ba, vec_bc)


@ms_function
def calc_angle_with_pbc(position_a: Tensor, position_b: Tensor, position_c: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Compute angle formed by three positions A-B-C at periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_c (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        angle (Tensor):  Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    # (B, ..., D)
    vec_ba = get_vector_with_pbc(position_b, position_a, pbc_box)
    vec_bc = get_vector_with_pbc(position_b, position_c, pbc_box)
    return calc_angle_between_vectors(vec_ba, vec_bc)


@ms_function
def calc_angle(position_a, position_b: Tensor, position_c: Tensor, pbc_box: Tensor = None) -> Tensor:
    r"""Compute angle formed by three positions A-B-C.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_c (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        angle (Tensor):  Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    # (B, ..., D)
    if pbc_box is None:
        vec_ba = get_vector_without_pbc(position_b, position_a)
        vec_bc = get_vector_without_pbc(position_b, position_c)
    else:
        vec_ba = get_vector_with_pbc(position_b, position_a, pbc_box)
        vec_bc = get_vector_with_pbc(position_b, position_c, pbc_box)
    return calc_angle_between_vectors(vec_ba, vec_bc)


@ms_function
def calc_torsion_for_vectors(vector1: Tensor, vector2: Tensor, vector3: Tensor) -> Tensor:
    r"""Compute torsion angle formed by three vectors.

    Args:
        vector1 (Tensor):   Tensor of shape (..., D). Data type is float.
        vector2 (Tensor):   Tensor of shape (..., D). Data type is float.
        vector3 (Tensor):   Tensor of shape (..., D). Data type is float.

    Returns:
        torsion (Tensor):   Tensor of shape (..., 1). Data type is float.

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    """

    # (B, ..., D) <- (B,...,1)
    v2norm = keep_norm_last_dim(vector2)
    # (B, ..., D) = (B, ..., D) / (...,1)
    norm_vec2 = vector2 / v2norm

    # (B, ..., D)
    vec_a = msnp.cross(norm_vec2, vector1)
    vec_b = msnp.cross(vector3, norm_vec2)
    cross_ab = msnp.cross(vec_a, vec_b)

    # (B,...,1)
    sin_phi = keepdim_sum(cross_ab*norm_vec2, -1)
    cos_phi = keepdim_sum(vec_a*vec_b, -1)

    return F.atan2(-sin_phi, cos_phi)


@ms_function
def calc_torsion_without_pbc(position_a: Tensor,
                             position_b: Tensor,
                             position_c: Tensor,
                             position_d: Tensor
                             ) -> Tensor:
    r"""Compute torsion angle formed by four positions A-B-C-D without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape (..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (..., D). Data type is float.
        position_c (Tensor):    Tensor of shape (..., D). Data type is float.
        position_d (Tensor):    Tensor of shape (..., D). Data type is float.

    Returns:
        torsion (Tensor):   Tensor of shape (..., 1). Data type is float.

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    """

    vec_ba = get_vector_without_pbc(position_b, position_a)
    vec_cb = get_vector_without_pbc(position_c, position_b)
    vec_dc = get_vector_without_pbc(position_d, position_c)
    return calc_torsion_for_vectors(vec_ba, vec_cb, vec_dc)


@ms_function
def calc_torsion_with_pbc(position_a: Tensor,
                          position_b: Tensor,
                          position_c: Tensor,
                          position_d: Tensor,
                          pbc_box: Tensor
                          ) -> Tensor:
    r"""Compute torsion angle formed by four positions A-B-C-D at periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_c (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_d (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        torsion (Tensor):   Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    vec_ba = get_vector_with_pbc(position_b, position_a, pbc_box)
    vec_cb = get_vector_with_pbc(position_c, position_b, pbc_box)
    vec_dc = get_vector_with_pbc(position_d, position_c, pbc_box)
    return calc_torsion_for_vectors(vec_ba, vec_cb, vec_dc)


@ms_function
def calc_torsion(position_a: Tensor,
                 position_b: Tensor,
                 position_c: Tensor,
                 position_d: Tensor,
                 pbc_box: Tensor = None
                 ) -> Tensor:
    r"""Compute torsion angle formed by four positions A-B-C-D.

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_c (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_d (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        torsion (Tensor):   Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Dimension of the simulation system. Usually is 3.

    """

    if pbc_box is None:
        vec_ba = get_vector_without_pbc(position_b, position_a)
        vec_cb = get_vector_without_pbc(position_c, position_b)
        vec_dc = get_vector_without_pbc(position_d, position_c)
    else:
        vec_ba = get_vector_with_pbc(position_b, position_a, pbc_box)
        vec_cb = get_vector_with_pbc(position_c, position_b, pbc_box)
        vec_dc = get_vector_with_pbc(position_d, position_c, pbc_box)

    return calc_torsion_for_vectors(vec_ba, vec_cb, vec_dc)


@ms_function
def get_kinetic_energy(mass: Tensor, velocity: Tensor) -> Tensor:
    r"""Compute kinectic energy of the simulation system.

    Args:
        mass (Tensor):      Tensor of shape (B, A). Data type is float.
                            Mass of the atoms in system.
        velocity (Tensor):  Tensor of shape (B, A, D). Data type is float.
                            Velocities of the atoms in system.

    Returns:
        kinectics (Tensor): Tensor of shape (B). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation
        A:  Number of atoms in the simulation system
        D:  Dimension of the simulation system. Usually is 3.

    """

    # (B, A) <- (B, A, D)
    v2 = F.reduce_sum(velocity*velocity, -1)
    # (B, A) * (B, A)
    kinectics = 0.5 * mass * v2
    # (B) <- (B, A)
    return F.reduce_sum(kinectics, -1)


def get_integer(value) -> int:
    r"""get integer type of the input value

    Args:
        value (Union[int, Tensor, Parameter, ndarray]): Input value

    Returns:
        int_value (int)

    """
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.asnumpy()
    return int(value)


def get_ndarray(value, dtype: type = None) -> np.ndarray:
    r"""get ndarray type of the input value

    Args:
        value (Union[Tensor, Parameter, ndarray]):  Input value
        dtype (type):                               Data type. Default: None

    Returns:
        int_value (int)

    """
    if value is None:
        return None
    if isinstance(value, (Tensor, Parameter)):
        value = value.asnumpy()
        if dtype is not None:
            value = value.astype(dtype)
    else:
        value = np.array(value, dtype)
    return value
