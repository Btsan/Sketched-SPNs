# Adapted from the kwisehash module by Heddes et al.
# https://github.com/mikeheddes/fast-multi-join-sketch (accessed Jan 2025)

import numpy as np
from numpy import int64

import torch

MERSENNE_PRIME = (1 << 61) - 1

def _poly_call(input:int64, seeds:int64, k:int) -> int64:
    # input has any shape
    # seeds has shape [m, k]
    new_shape = seeds.shape + (1,) * input.ndim
    seeds = seeds.reshape(new_shape)

    output = seeds[:, 0]
    for i in range(1, k):
        tmp0 = output * input + seeds[:, i]

        ### either modulo
        # output = tmp0 % MERSENNE_PRIME

        ### or bitwise AND with range check
        tmp1 = (tmp0 & MERSENNE_PRIME) + (tmp0 >> 61)
        # tmp2 = tmp1 - MERSENNE_PRIME
        # output = np.where(tmp2 < 0, tmp1, tmp2)
        tmp1[tmp1 >= MERSENNE_PRIME] -= MERSENNE_PRIME
        output = tmp1

        ### everything in one line
        # output = (output * input + seeds[:, i]) % MERSENNE_PRIME
    return output

class SignHash(object):
    def __init__(self, depth: int, k: int = 4):
        a = np.random.randint(1, MERSENNE_PRIME, (depth, k-1))
        b = np.random.randint(0, MERSENNE_PRIME, (depth, 1))
        self.seeds = np.concatenate([a, b], axis=1)
        self.depth = depth
        self.k = k
    
    def __call__(self, input:int64):
        input = np.array(input).astype(int64)
        output = _poly_call(input, self.seeds, self.k)
        output = (output & 1) * 2 - 1
        return torch.tensor(output, dtype=torch.long)
    
class BinHash(object):
    def __init__(self, depth: int, width: int, k: int = 2):
        a = np.random.randint(1, MERSENNE_PRIME, (depth, k-1))
        b = np.random.randint(0, MERSENNE_PRIME, (depth, 1))
        self.seeds = np.concatenate([a, b], axis=1)
        self.depth = depth
        self.width = width
        self.k = k

    def __call__(self, input:int64):
        input = np.array(input).astype(int64)
        output = _poly_call(input, self.seeds, self.k)
        output %= self.width
        return torch.tensor(output, dtype=torch.long)

if __name__ == '__main__':
    from time import perf_counter

    input = np.linspace(0, MERSENNE_PRIME - 1, (1<<20) - 1, dtype=int64)[1:]
    print(f"input shape {input.shape} ({input.dtype})")

    iterations = 10
    depth = 5
    width = 10000
    K = 64
    xi = SignHash(depth, k=K)
    b_xi = BinHash(depth, width, k=K)

    t0 = perf_counter()
    signs = sum([xi(input) for _ in range(iterations)])
    t1 = perf_counter()
    print(f"[{(t1-t0)*1000/iterations:,.2f} ms] Mean sign = {signs.to(float).mean():,.2e} (Expectation 0)")
    
    t0 = perf_counter()
    bins = sum([b_xi(input) for _ in range(iterations)]) / iterations
    t1 = perf_counter()
    print(f"[{(t1-t0)*1000/iterations:,.2f} ms] Mean bin = {bins.to(float).mean():,.2f} (Expectation {width/2:,.1f})")

    
    a = np.random.randint(1, MERSENNE_PRIME, (depth, K-1))
    b = np.random.randint(0, MERSENNE_PRIME, (depth, 1))
    seeds = np.concatenate([a, b], axis=1)

    t0 = perf_counter()
    permute = sum([_poly_call(input, seeds, k=K) for _ in range(iterations)])
    t1 = perf_counter()
    print(f"timing results = {(t1-t0)*1000/iterations:,.2f} ms")



    