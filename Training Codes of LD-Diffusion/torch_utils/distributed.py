# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
import warnings
from . import training_stats

#----------------------------------------------------------------------------

def init():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    # Select backend: allow override via env, otherwise choose GLOO for single-process or CPU, NCCL for multi-GPU.
    backend_env = os.environ.get('TORCH_DISTRIBUTED_BACKEND', None)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if backend_env in ('nccl', 'gloo'):
        backend = backend_env
    else:
        if os.name == 'nt':
            backend = 'gloo'
        else:
            backend = 'nccl' if (torch.cuda.is_available() and world_size > 1) else 'gloo'

    # Skip init_process_group if world_size is 1 to avoid Gloo errors on Windows single-process
    if world_size == 1:
        return

    try:
        torch.distributed.init_process_group(backend=backend, init_method='env://')
    except Exception as e:
        if backend == 'nccl':
            warnings.warn(f"NCCL backend init failed ({e}); falling back to GLOO.")
            torch.distributed.init_process_group(backend='gloo', init_method='env://')
        else:
            raise

    # Set CUDA device only when CUDA is available and NCCL is used.
    if backend == 'nccl' and torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        except Exception as e:
            warnings.warn(f"torch.cuda.set_device failed ({e}); continuing without setting device.")

    sync_device = torch.device('cuda') if (get_world_size() > 1 and torch.cuda.is_available() and backend == 'nccl') else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

#----------------------------------------------------------------------------

def should_stop():
    return False

#----------------------------------------------------------------------------

def update_progress(cur, total):
    _ = cur, total

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------
