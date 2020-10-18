import random

from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from scipy import stats

DTYPE = torch.float32

class UltraDense(nn.Module):
    def __init__(self, embedding_size, tgt_comp):
        """

        :param embedding_size: Integer, the dimension of embedding space
        :param tgt_comp: Integer, the target_component for ultradense space
        """
        super().__init__()

        self.d = embedding_size
        self.Pc = np.zeros(embedding_size)
        self.Pc[tgt_comp] = 1
        self.Pc = torch.tensor(self.Pc, dtype=DTYPE)

        # This should remain orthogonal
        self.Q = nn.Parameter(torch.randn((embedding_size, embedding_size), dtype=DTYPE) * 0.01)

    def apply_q(self, embeddings):
        return torch.matmul(torch.tensor(embeddings, dtype=DTYPE), self.Q).detach().numpy()

    def forward(self, embeddings, labels):

        """
        Runs the forward pass of the model.

        :param embeddings: real numbers of shape (batch_size, embedding_size)
        :param labels: 1 or 0 (in or out of class), (batch_size, embedding_size)

        :return: Lcts result (batch_size / 2, 1), Lct result (batch_size / 2, 1)
        """
        
        # Establishing the sets Lc~ and Lc~/
        num_ones = sum(labels)
        num_zeroes = len(labels) - num_ones
        
        assert num_zeroes == num_ones
        assert num_zeroes % 2 == 0 # Is even
        
        sort_lbl_idx = labels.argsort()
        
        # Lcts
        u_idces = sort_lbl_idx[:num_zeroes] # 0-indices
        v_idces = sort_lbl_idx[num_zeroes:] # 1-indicies
        
        u_batch_lcts = embeddings[u_idces]
        v_batch_lcts = embeddings[v_idces]
        Lcts_batch_diff = u_batch_lcts - v_batch_lcts
        Lcts_batch_diff = torch.tensor(Lcts_batch_diff, dtype=DTYPE)
        
        Lcts_linear = torch.matmul(Lcts_batch_diff, self.Q)
        Lcts_result = torch.matmul(Lcts_linear, self.Pc.T)
            
        # Lct
        u_idces_a = u_idces[:num_zeroes//2]
        u_idces_b = u_idces[num_zeroes//2:]
        v_idces_a = v_idces[:num_ones//2]
        v_idces_b = v_idces[num_ones//2:]
        
        u_batch_a_lct = embeddings[u_idces_a]
        u_batch_b_lct = embeddings[u_idces_b]
        v_batch_a_lct = embeddings[v_idces_a]
        v_batch_b_lct = embeddings[v_idces_b]
        
        Lct_u_diff = u_batch_a_lct - u_batch_b_lct
        Lct_v_diff = v_batch_a_lct - v_batch_b_lct
        Lct_u_diff = torch.tensor(Lct_u_diff, dtype=DTYPE)
        Lct_v_diff = torch.tensor(Lct_v_diff, dtype=DTYPE)
        
        Lct_batch_diff = torch.cat((Lct_u_diff, Lct_v_diff))

        
        Lct_linear = torch.matmul(Lct_batch_diff, self.Q)
        Lct_result = torch.matmul(Lct_linear, self.Pc.T)
        
        return Lcts_result, Lct_result

    def loss_func(self, Lcts_result, Lct_result):
        return torch.sum(torch.abs(Lct_result)) - torch.sum(torch.abs(Lcts_result))

    def orthogonalize(self):
        u, s, vh = np.linalg.svd(self.Q.detach().numpy())
        self.Q.data = torch.tensor(np.matmul(u, vh))


if __name__ == "__main__":

    embed_size = 20
    batch_size = 20

    labels = [1] * (batch_size // 2) + [0] * (batch_size // 2)
    random.shuffle(labels)
    labels = np.array(labels)

    embeddings = np.random.random((batch_size, embed_size))

    embeddings[:,0] = labels

    offset_choice = 3
    model = UltraDense(embed_size, offset_choice)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print("Initial embeddings", embeddings[:,offset_choice])
    print("Labels", labels)

    num_epochs = 10
    for e in range(num_epochs):
        optimizer.zero_grad()
    
        rand_indices = [i for i in range(batch_size)]
        random.shuffle(rand_indices)
    
        Lcts_result, Lct_result = model(embeddings[rand_indices], labels[rand_indices])
        loss = model.loss_func(Lcts_result, Lct_result)
        print("loss", loss)
        loss.backward()
        optimizer.step()
    
        #print("Q before ortho:", model.Q)
    
        model.orthogonalize()
    
        #print("Q after ortho:", model.Q)
    
        output_ultradense = model.apply_q(embeddings)[:,offset_choice]
        #print("Epoch", e, output_ultradense)
        sorted_result_idx = np.argsort(output_ultradense)

        #print("Sorted component", output_ultradense[sorted_result_idx])
        rval, pval = stats.spearmanr(labels, output_ultradense)
        print("Correlation", rval)

