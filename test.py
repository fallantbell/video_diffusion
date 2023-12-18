import torch


dim_mults=(1, 2, 4, 8)
init_dim = 64
dim = 64

dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

print(dims)
in_out = list(zip(dims[:-1], dims[1:]))

print(in_out)