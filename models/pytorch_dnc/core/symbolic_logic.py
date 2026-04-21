import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import json
import copy
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               stride_row, # *Pointer* to second input vector.
               n_rows,n_cols,  # *Pointer* to output vector.
               BLOCK_SIZE_M: tl.constexpr, 
               BLOCK_SIZE_N: tl.constexpr 
               ):
    
    pid_m = tl.program_id(axis=0)
    #pid_n = tl.program_id(axis=0)
    #print("pid_m",pid_m)
    #print("pid_n",pid_n)
    #print("stride_row",stride_row)

    rm = pid_m * BLOCK_SIZE_M*BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_M*BLOCK_SIZE_N)
    #rn = pid_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)

    #offsets = rm[:,None] * stride_row + rn[None,:]
    offsets = rm[:,None] # + rn[None,:]
    #print("offsets",offsets)
    #print("rm",rm)
    mask = (rm[:,None] < n_rows*n_cols) # & (rn[None,:] < n_cols)
    #print("mask",mask)
    x = tl.load(x_ptr + offsets)#, mask=mask)
    output = x + 0.2
    tl.store(x_ptr + offsets, output)#, mask=mask)

@torch.compile
def add_one(input):
    output = copy.deepcopy(input) #:)
    for i in range(len(input)):
       for j in range(len(input[i])):
            if input[i][j]<9:
                output[i][j] +=1
    return output

@torch.compile
def complex_1(input):
    output = copy.deepcopy(input) #:)
    for i in range(len(input)):
       for j in range(len(input[i])):
            input[i][j] = int((input[i][j]**2)/5)
    for i in range(len(input)):
       for j in range(len(input[i])):
           if input[i][j] >  9:
               input[i][j] = 9
           if input[i][j] <  0:
               input[i][j] = 0
    return output


class SymbolicLogic(nn.Module):
    def __init__(self):
        super(SymbolicLogic, self).__init__()
        self.BLOCK_SIZE = 8 

    
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

        n_elements = memory_vb.numel()
        
        #grid = lambda meta: True#((n_elements), )
        grid = lambda meta:(triton.cdiv(n_elements, meta["BLOCK_SIZE_N"]*meta["BLOCK_SIZE_M"]),)
        
        memory_vb_symbolic_old = memory_vb.clone()
        #print("grid",triton.cdiv(n_elements, 8*8))
        triton_kernel=add_kernel[grid](memory_vb, stride_row=8,n_cols = 8,n_rows=8, BLOCK_SIZE_M=8,BLOCK_SIZE_N=8)
        #print("memory_vb",memory_vb-memory_vb_symbolic_old)

        torch.cuda.synchronize()
        # memory_vb_symbolic_old = memory_vb_symbolic.clone()
        #print("memory_vb_symbolic.shape",memory_vb_symbolic.shape)
        #for i in range(len(memory_vb)):
        #    memory_vb[i] = add_one(memory_vb[i])    #self.soduko_solver_v1(memory_vb_symbolic)
        
        # for i in range(memory_vb_symbolic.shape[0]):
        #      for j in range(memory_vb_symbolic.shape[1]):
        #          if memory_vb_symbolic[i][j] != memory_vb_symbolic_old[i][j]:
        #              memory_vb[i][j] = F.one_hot(memory_vb_symbolic[i][j], num_classes=10).float()   
        # here we simply concatenate the hidden state and the read vector
        return memory_vb
    
    def generate_challenge(self,height,width):
        input =  [[]]*width
        
        function_list = [add_one,complex_1]

        for i in range(len(input)):
            input[i] = [0]*height

        for i in range(len(input)):
            for j in range(len(input[0])):
                input[i][j] = random.randint(0,9)
        random_function = random.choice(function_list)
        return input , random_function(input)
    
    def generate_symbolic_dataset(self,name,num_examples=100):
        
        num_train_examples = 2
        num_test_examples = 1
        training_dataset = {}
        training_solutions_dataset = {}

        for n in range(num_examples):
            example_name = ""
            for i in range(8):
                example_name += str(random.randint(0,9))
            train = []
            test = []
            solutions = []
            
            for i in range(num_train_examples):
                input ,output =  self.generate_challenge(random.randint(5,8),random.randint(5,8))
                train.append({"input":input, "output":output})
            
            for i in range(num_test_examples):
                input ,output =  self.generate_challenge(random.randint(5,8),random.randint(5,8))
                test.append({"input":input})
                solutions.append(output)
            training_dataset[example_name] = {"train":train,"test":test}
            training_solutions_dataset[example_name] = solutions
        with open(name+'_challenges.json', 'w') as f:
           json.dump(training_dataset, f)
        with open(name+'_solutions.json', 'w') as f:
           json.dump(training_solutions_dataset, f)

def generate_dataset():
    sl = SymbolicLogic()
    sl.generate_symbolic_dataset("arc-agi_trainingSym")
    sl.generate_symbolic_dataset("arc-agi_evaluationSym")
    sl.generate_symbolic_dataset("arc-agi_conceptSym")
if __name__ == "__main__":
    generate_dataset()