import contextlib
import os
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda


@contextlib.contextmanager
def temp_numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def set_seed(local_rank):
    # os.environ['PYTHONHASHSEED'] = str(0)
    random.seed(0 + local_rank)  # python seed
    np.random.seed(0 + local_rank)  # seed the global NumPy RNG
    torch.manual_seed(4 + local_rank)  # PyTorch random number generator
    # torch.cuda.manual_seed_all(0)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # CUDA convolution benchmarking


'''DataLoader will reseed workers following Randomness in multi-process
   data loading algorithm. Use worker_init_fn() and generator to preserve 
   reproducibility'''
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

dataloader_generator = torch.Generator()
dataloader_generator.manual_seed(0)

'''
torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    num_workers=8,
    worker_init_fn=seed_worker,
    generator=g,
)'''
