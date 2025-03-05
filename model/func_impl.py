import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    fc_layer: str,
    in_dim: int,
    out_dim: int,
):
    """
    Prepare necessary information for later communications in forward and backward passes.

    Parameters
    ----------
    comm : Communicator
        The global MPI communicator.
    rank : int
        The global rank of the process.
    mp_size : int
        Model Parallel size.
    dp_size : int
        Data Parallel size.
    fc_layer : str
        Identifier for the fully-connected layer. It must be one of:
        'fc_q', 'fc_k', 'fc_v', or 'fc_o'.
        - For 'fc_q', 'fc_k', and 'fc_v', the partitioning is along the output dimension.
        - For 'fc_o', the partitioning is along the input dimension.
    in_dim : int
        Original input feature dimension.
    out_dim : int
        Original output feature dimension.

    Returns
    -------
    mp_idx : int
        Model parallel index (position within a data parallel replica).
    dp_idx : int
        Data parallel index (which replica this process belongs to).
    mp_comm : Communicator
        The model parallel communicator (all processes in one data parallel replica).
    dp_comm : Communicator
        The data parallel communicator (all processes holding the same weight shard).
    part_in_dim : int
        The partitioned input dimension for the FC layer.
    part_out_dim : int
        The partitioned output dimension for the FC layer.
    """
    
    mp_idx = rank % mp_size  # part of model 
    dp_idx = rank // mp_size # copy of model

    mp_comm = comm.Split(color=dp_idx, key=rank)  # for same DP group
    dp_comm = comm.Split(color=mp_idx, key=rank)  # for same MP shard

    if fc_layer in ['fc_q', 'fc_k', 'fc_v']:  
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size
    elif fc_layer == 'fc_o':  
        part_in_dim = in_dim // mp_size
        part_out_dim = out_dim 

    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim

def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward inputs from all model-parallel nodes.

    Each node holds a piece of the full input with shape:
      (batch_size, seq_length, part_in_dim)
    After gathering, the full input should have shape:
      (batch_size, seq_length, part_in_dim * mp_size)
    """
    batch_size, seq_length, part_in_dim = x.shape

    x = np.ascontiguousarray(x)
    temp_buffer = np.empty((mp_size, batch_size, seq_length, part_in_dim), dtype=x.dtype)
    mp_comm.Allgather(x, temp_buffer)
    collected_x = np.concatenate(np.ascontiguousarray(temp_buffer), axis=-1)  # (batch_size, seq_length, part_in_dim * mp_size)

    return collected_x

def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward outputs from all model-parallel nodes.

    Each node holds a piece of the full output with shape:
      (batch_size, seq_length, part_out_dim)
    After gathering, the full output should have shape:
      (batch_size, seq_length, part_out_dim * mp_size)
    """
    batch_size, seq_length, part_out_dim = out.shape

    out = np.ascontiguousarray(out) # make contiguous array 
    temp_buffer = np.empty((mp_size, batch_size, seq_length, part_out_dim), dtype=out.dtype)
    mp_comm.Allgather(out, temp_buffer)
    collected_out = np.concatenate(np.ascontiguousarray(temp_buffer), axis=-1)  # (batch_size, seq_length, out_dim)

    return collected_out

def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """
    Collect the fc output layer's output gradient for the local MP node.
    
    In our setup, the full output_grad is a 3-D tensor of shape 
        (batch_size, seq_length, out_dim),
    and the fully connected layer's weight is partitioned along out_dim.
    Therefore, we split output_grad along axis=2 into mp_size parts and
    return the part corresponding to mp_group_idx.
    
    Parameters
    ----------
    output_grad : np.ndarray
        The full output gradient from fc_o with shape 
        (batch_size, seq_length, out_dim).
    mp_group_idx : int
        The current model parallel node's index.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_output_grad : np.ndarray
        The local output gradient for this MP node with shape 
        (batch_size, seq_length, out_dim // mp_size).
    """
    part_out_dim = output_grad.shape[2] // mp_size  # Divide among MP nodes

    collected_output_grad = output_grad[:, :, mp_group_idx * part_out_dim : (mp_group_idx + 1) * part_out_dim]

    return collected_output_grad


def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Use reduce-scatter / all-to-all to combine the contributions for grad_x from all nodes
    and scatter the reduced result along the input feature dimension.
    
    The grad_x tensor (gradient with respect to fc_o's input) has shape
        (batch_size, seq_length, in_dim),
    and the fc_o's weight matrix is sharded along the in_dim axis. In the 
    backward pass, each node computes a local grad_x and then these must be 
    summed across nodes. Instead of summing the full tensor and then slicing,
    we perform a reduce-scatter / all-to-all.
    
    Parameters
    ----------
    grad_x : np.ndarray
        The locally computed grad_x for fc_o, of shape 
        (batch_size, seq_length, in_dim).
    mp_comm :
        The model parallel communicator. It is assumed to expose methods such as reduce-scatter / all-to-all.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_grad_x : np.ndarray
        The reduced and scattered grad_x with shape 
        (batch_size, seq_length, in_dim // mp_size).
    """
    B, S, I = grad_x.shape
    rank = mp_comm.Get_rank()

    # sum across all ranks
    sum_buf = np.empty_like(grad_x)
    mp_comm.Allreduce(grad_x, sum_buf, op=MPI.SUM)

    chunk = I // mp_size
    start_col = rank * chunk
    end_col = (rank + 1) * chunk
    collected_grad_x = sum_buf[:, :, start_col:end_col]

    return collected_grad_x