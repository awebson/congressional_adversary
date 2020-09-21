

import random

import torch
import torch.nn as nn
import numpy as np

use_dtype = torch.float32

def get_norm_sample(length, scale=0.01):
    return torch.randn(length, dtype=use_dtype) * scale

class OrthoBasis(nn.Module):

    def __init__(self, vec_size, dev, delta=0.4):
        super(OrthoBasis, self).__init__()
        self.dev = dev
        self.vec_size = vec_size
        self.delta_size = int(delta * vec_size)
        self.gamma_size = vec_size - self.delta_size

        self.blendgen = nn.Parameter(torch.randn((vec_size, vec_size), dtype=use_dtype) * 0.01)
        self.H = nn.Parameter(torch.randn((vec_size, vec_size), dtype=use_dtype) * 0.01)

    def forward(self, encoding_batch):
    
    
        rotated = torch.matmul(encoding_batch, self.H)
        blend_input = torch.matmul(rotated, self.blendgen)
        blend_c = torch.sigmoid(blend_input)
        blend_d = 1 - blend_c
        
        c_vecs = rotated * blend_c
        d_vecs = rotated * blend_d
        
        return c_vecs, d_vecs
    
    def orthogonalize(self):
        u, s, vh = np.linalg.svd(self.H.detach().numpy())
        self.H = nn.Parameter(torch.tensor(np.matmul(u, vh))).to(self.dev)
        

if __name__ == "__main__":
    
    batch_size = 2
    vec_length = 16
    num_epochs = 50
    
    dev = torch.device('cpu')
    
    input_batch = get_norm_sample((batch_size,vec_length), scale=1.0)
    tgt_delta_batch = get_norm_sample((batch_size,vec_length), scale=1.0)
    tgt_gamma_batch = get_norm_sample((batch_size,vec_length), scale=1.0)
    cosine_label = torch.ones(batch_size, vec_length)
    
    model = OrthoBasis(vec_length, dev)
    optimizer = torch.optim.Adam(model.parameters())
    
    def loss_fn(a, b):
        return 1 - nn.functional.cosine_similarity(a, b, dim=-1).mean()

    model = model.train()
    for e in range(num_epochs):
        optimizer.zero_grad()
        gamma_vec, delta_vec = model(input_batch)
        loss_delta = loss_fn(tgt_delta_batch, delta_vec)
        loss_gamma = loss_fn(tgt_gamma_batch, gamma_vec)
        loss_joint = loss_delta + loss_gamma
        loss_joint.backward()
        optimizer.step()
        model.orthogonalize()
    
        print("cos sim of result", nn.functional.cosine_similarity(
            gamma_vec + delta_vec,
            input_batch,
            dim=-1).mean())
    
        #print("gen_vec", gen_vec.tolist())
        print("loss", float(loss_joint))

    print("gamma_vec", gamma_vec, "delta_vec", delta_vec)
    print("sum", delta_vec + gamma_vec)
    print("input_batch", input_batch)
