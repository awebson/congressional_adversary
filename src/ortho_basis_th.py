

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

        self.vec_size = vec_size
        self.delta_size = int(delta * vec_size)
        self.gamma_size = vec_size - self.delta_size

        self.x1_1 = nn.Parameter(get_norm_sample(1))
        self.x1_2 = nn.Parameter(get_norm_sample(1))
        self.y_1 = nn.Parameter(get_norm_sample(self.vec_size - 1))
        self.y_2 = nn.Parameter(get_norm_sample(self.vec_size - 1))
        
        self.I = torch.tensor(np.identity(self.vec_size - 1), dtype=use_dtype).to(dev)
        self.t1 = torch.tensor(1.0).to(dev)
        self.t1overVc = torch.tensor((1 / self.vec_size,)).to(dev)

    def forward(self, encoding_batch):
    
        def constrain_x1(x1_val):
            return self.t1overVc * torch.tanh(x1_val) * (
                self.t1 - torch.tanh(x1_val) * torch.tanh(x1_val)
            )
        
        y_len = self.vec_size - 1

        def generate_h(x1_val, y_val):
            n = torch.norm(y_val) / torch.sqrt(1.0 - x1_val) # What to divide self.y by to get norm
            y_tilde = y_val / n
            u1_tilde_flat = torch.cat((x1_val, y_tilde))
            u1_tilde = u1_tilde_flat.reshape((self.vec_size, 1))

            Y = u1_tilde[1:]
            YT = Y.reshape((1, y_len))
        
            bot_right = self.I - (1.0 / (1 - x1_val)) * torch.matmul(Y, YT)

            right = torch.cat((YT, bot_right))
            return torch.cat((u1_tilde, right), dim=1) 

        x1_1 = constrain_x1(self.x1_1)
        x1_2 = constrain_x1(self.x1_2)

        H_1 = generate_h(x1_1, self.y_1)
        H_2 = generate_h(x1_2, self.y_2)
        
        H = torch.matmul(H_1, H_2)
        H_T = torch.transpose(H, 0, 1)

        coeffs = torch.matmul(H_T, torch.transpose(encoding_batch, 0, 1))
        
        coeffs_C = coeffs[:self.gamma_size,:]
        coeffs_D = coeffs[self.gamma_size:,:]
        
        H_C = H[:,:self.gamma_size]
        H_D = H[:,self.gamma_size:]
        
        c_vecs = torch.transpose(torch.matmul(H_C, coeffs_C), 0, 1)
        d_vecs = torch.transpose(torch.matmul(H_D, coeffs_D), 0, 1)
        
        return c_vecs, d_vecs
        

if __name__ == "__main__":
    
    batch_size = 2
    vec_length = 16
    num_epochs = 50
    
    input_batch = get_norm_sample((batch_size,vec_length), scale=1.0)
    tgt_delta_batch = get_norm_sample((batch_size,vec_length), scale=1.0)
    tgt_gamma_batch = get_norm_sample((batch_size,vec_length), scale=1.0)
    cosine_label = torch.ones(batch_size, vec_length)
    
    model = OrthoBasis(vec_length)
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
    
        print("cos sim of result", nn.functional.cosine_similarity(
            gamma_vec + delta_vec,
            input_batch,
            dim=-1).mean())
    
        #print("gen_vec", gen_vec.tolist())
        print("loss", float(loss_joint))

    print("gamma_vec", gamma_vec, "delta_vec", delta_vec)
    print("sum", delta_vec + gamma_vec)
    print("input_batch", input_batch)
