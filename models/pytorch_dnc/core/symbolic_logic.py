import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class SymbolicLogic(nn.Module):
    def __init__(self):
        super(SymbolicLogic, self).__init__()

    
    def soduko_solver_v1(self, memory_vb_symbolic):
        for i in range(memory_vb_symbolic.shape[0]):
            number_index_list = []
            numbers_present = []
            counter = 0
            for j in range(10):
                if j in memory_vb_symbolic[i]:
                    number_index_list.append(memory_vb_symbolic[i].tolist().index(j))
                    numbers_present.append(j)   
            for j in range(10):
                counter += 1
                if j in numbers_present:
                    continue
                if counter >=10:
                    break
                memory_vb_symbolic[i][counter] = j
        return memory_vb_symbolic
    def forward(self, memory_vb):
        
        memory_vb_symbolic = torch.argmax(memory_vb, dim=-1)
        memory_vb_symbolic_old = memory_vb_symbolic.clone()
        memory_vb_symbolic = self.soduko_solver_v1(memory_vb_symbolic)
        for i in range(memory_vb_symbolic.shape[0]):
            for j in range(memory_vb_symbolic.shape[1]):
                if memory_vb_symbolic[i][j] != memory_vb_symbolic_old[i][j]:
                    memory_vb[i][j] = F.one_hot(memory_vb_symbolic[i][j], num_classes=10).float()   
        # here we simply concatenate the hidden state and the read vector
        return memory_vb