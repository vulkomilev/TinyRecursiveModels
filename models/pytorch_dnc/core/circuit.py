from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import visdom
import matplotlib.pyplot as plt 
import copy
from functools import reduce
import wandb
BATCH_SIZE = 80     
torch.autograd.set_detect_anomaly(True)
class Circuit(nn.Module):   # NOTE: basically this whole module is treated as a custom rnn cell
    def __init__(self, args):
        super(Circuit, self).__init__()
        # logging
        self.logger = args.logger
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        # params
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.num_write_heads = args.num_write_heads
        self.num_read_heads = args.num_read_heads
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value
        self.img_counter = 0
        self.input_vb = 'input_vb'
        #self.out_fix = torch.nn.Linear(in_features=819200,out_features=409600)
        # functional components
        self.controller_params = args.controller_params
        self.accessor_params = args.accessor_params
        self.vis = visdom.Visdom()
        # now we fill in the missing values for each module
        self.read_vec_dim = self.num_read_heads * self.mem_wid
        # controller
        self.controller_params.batch_size = self.batch_size
        self.controller_params.input_dim = self.input_dim
        self.controller_params.read_vec_dim = self.read_vec_dim
        self.controller_params.output_dim = self.output_dim
        self.controller_params.hidden_dim = self.hidden_dim
        self.controller_params.mem_hei = self.mem_hei
        self.controller_params.mem_wid = self.mem_wid
        self.controller_params.clip_value = self.clip_value
        # accessor: {write_heads, read_heads, memory}
        self.accessor_params.batch_size = self.batch_size
        self.accessor_params.hidden_dim = self.hidden_dim
        self.accessor_params.num_write_heads = self.num_write_heads
        self.accessor_params.num_read_heads = self.num_read_heads
        self.accessor_params.mem_hei = self.mem_hei
        self.accessor_params.mem_wid = self.mem_wid
        self.accessor_params.clip_value = self.clip_value
        self.mask_input = torch.nn.Linear(64*BATCH_SIZE,64*BATCH_SIZE)

        self.init_wandb = False

        self.logger.warning("<-----------------------------======> Circuit:    {Controller, Accessor}")

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def print_model(self):
        self.logger.warning("<-----------------------------======> Circuit:    {Overall Architecture}")
        self.logger.warning(self)

    def _reset_states(self): # should be called at the beginning of forwarding a new input sequence
        # we first reset the previous read vector
        self.read_vec_vb = Variable(self.read_vec_ts).type(self.dtype)
        # we then reset the controller's hidden state
        self.controller._reset_states()
        # we then reset the write/read weights of heads
        self.accessor._reset_states()
        

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()
        # reset internal states
        self.read_vec_ts = torch.zeros(self.batch_size, self.num_read_heads,self.output_dim).fill_(1e-6)
        self._reset_states()
       
    def reset_visual(self):
        pass
        #self.accessor.reset_visual()
    @torch._dynamo.disable
    def tricky_numpy_logic(self,data):
         plt.figure(figsize=(12, 8))
         plt.imshow(data.clone().cpu().float().numpy().tolist())
         plt.savefig("./image_data/"+str(self.img_counter)+".png",dpi=300)
         self.img_counter +=1
    #@torch.compile
    def soduko_solver_v1(self, memory_vb_symbolic):
            
            # Determine the number of rows
            num_rows = memory_vb_symbolic.shape[0]
            
            for i in range(num_rows):
                # Extract the row as a 1D tensor without converting to a Python list
                row = memory_vb_symbolic[i] 
                
                # Use PyTorch native operations to find what numbers are present
                # We convert it to a standard python set for lightning-fast 'in' checks
                # without triggering PyTorch computation graphs
                numbers_present = set(row.tolist()) 
                
                counter = 0
                for j in range(10):
                    counter += 1
                    if j in numbers_present:
                        continue
                    if counter >= 10:
                        break
                    
                    # In-place modification of the tensor
                    memory_vb_symbolic[i][counter] = j
                    
            return memory_vb_symbolic
    def soduko_solver_v1_no_loop(self, memory_vb_symbolic):
            
        
                # Extract the row as a 1D tensor without converting to a Python list
                row = memory_vb_symbolic[1] 
                
                # Use PyTorch native operations to find what numbers are present
                # We convert it to a standard python set for lightning-fast 'in' checks
                # without triggering PyTorch computation graphs
                numbers_present = row.tolist() 
                
                counter = 0
                for j in range(10):
                    counter += 1
                    if j in numbers_present:
                        continue
                    if counter >= 10:
                        break
                    
                    # In-place modification of the tensor
                    memory_vb_symbolic[1][counter] = j
                    
                return memory_vb_symbolic
    def аddd(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[2] +=1
                return memory_vb_symbolic
    def sub(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[3] -=1
                return memory_vb_symbolic
    def mul(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[4] *=2
                return memory_vb_symbolic
    def div(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[5] /=2
                return memory_vb_symbolic
    def abs(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[6] = torch.abs(memory_vb_symbolic[6])
                return memory_vb_symbolic
    def rev(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[7] = torch.flip(memory_vb_symbolic[7],dims=[0, 1])
                return memory_vb_symbolic
    def acos(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[8] = torch.acos(memory_vb_symbolic[8])
                return memory_vb_symbolic
    def acosh(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[9] = torch.acosh( memory_vb_symbolic[9])
                return memory_vb_symbolic
    def atan(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[10] = torch.atan( memory_vb_symbolic[10])
                return memory_vb_symbolic

    def mask_funct(self, memory_vb_symbolic):
           print("memory_vb_symbolic",memory_vb_symbolic.shape)
           memory_vb_symbolic = self.mask_input(memory_vb_symbolic)
           return memory_vb_symbolic

    @torch.compile  
    def forward_no_controller(self, input_vb):
        if not self.init_wandb:
            self.init_wandb = True
            print("wandb.watch(self.accessor)")
            #wandb.watch(self.accessor)
        
        input_vb_reshaped = torch.reshape(torch.clone(input_vb), ( -1,BATCH_SIZE* 64))
                #     #with torch.no_grad():
        #print(input_vb_reshaped.data[:80,:80].clone().cpu().float().numpy().shape)
        #print('point 1')
        #input_vb_reshaped.clone().cpu().float().numpy()
        
        #self.read_vec_vb = self.accessor.forward(input_vb_reshaped)
        shape_1 = input_vb.shape[0]
        shape_2 = input_vb.shape[1]
        #self.tricky_numpy_logic(input_vb_reshaped)
        
        #if self.training:
        #   with torch.no_grad():
              
       
        #      self.accessor.visual()
        #print("self.read_vec_vb.view(-1, self.read_vec_dim))",self.read_vec_vb.view(-1, self.read_vec_dim).expand(shape_1,shape_2, -1).shape)
        #print('self.read_vec_vb',self.read_vec_vb.shape)
        # torch.Size([80, 80, 64])
        #torch.cat((input_vb,
        #                                                torch.zeros_like(self.read_vec_vb.view(-1, self.read_vec_dim))), 1)
        #print("22222222222222",self.read_vec_vb.shape)
        #print("self.read_vec_dim",self.read_vec_dim)
        #print("input_vb",input_vb.shape) 
        #print("shape_1,shape_2",shape_1,shape_2)
        #print("self.read_vec_vb.view(-1, self.read_vec_dim)",self.read_vec_vb.view(-1, self.read_vec_dim).shape)

        #print("self.read_vec_vb.view(-1, self.read_vec_dim)",self.read_vec_vb.view(-1, self.read_vec_dim).expand(shape_1,shape_2, -1).shape)
        #print(torch.cat((input_vb,
        #                                                self.read_vec_vb.view(-1, self.read_vec_dim).expand(shape_1,shape_2, -1)), 2).shape)
        #print(torch.cat((input_vb,
        #                                                self.read_vec_vb.view(-1, self.read_vec_dim).reshape(shape_1,shape_2, -1)), 2).shape)

        #output_vb = self.hid_to_out(torch.cat((input_vb,
        #                                                self.read_vec_vb.view(-1, self.read_vec_dim).reshape(shape_1,shape_2, -1)), 2))
        mask_add  = self.mask_input(input_vb_reshaped) # self.mask_funct(input_vb_reshaped)
        solver_result_1 = self.soduko_solver_v1(input_vb_reshaped)
        solver_result_2 = self.аddd(solver_result_1)
        solver_result_3 = self.sub(solver_result_2)

        #solver_result_4 = self.mul(solver_result)
        #solver_result_5 = self.div(solver_result)
        #solver_result_6 = self.abs(solver_result)
#        solver_result_7 = self.rev(solver_result)
        #solver_result_8 = self.acos(input_vb_reshaped)
        #solver_result_9 = self.acosh(input_vb_reshaped)
        #solver_result_10 = self.atan(input_vb_reshaped)
        #print("input_vb",input_vb.shape)
        #print("solver_result",solver_result.shape)
        #print("self.hid_to_out",torch.cat((input_vb,solver_result.reshape(shape_1,shape_2, -1)),dim=1).shape)
        config = wandb.config
        
        # Use the parameters in your mock training loop
        #output_vb = torch.add(input_vb,solver_result_1.reshape(shape_1,shape_2, -1),alpha=1)#.view(-1, self.hidden_dim))
        #output_vb = torch.add(output_vb,solver_result_8.reshape(shape_1,shape_2, -1),alpha=1)
        #output_vb = torch.add(output_vb,solver_result_9.reshape(shape_1,shape_2, -1),alpha=1)
        #output_vb = self.hid_to_out(torch.add(input_vb,solver_result_10.reshape(shape_1,shape_2, -1),alpha=1))
        #output_vb = self.hid_to_out(input_vb)#.view(-1, self.hidden_dim))
        #output_vb = self.hid_to_out(solver_result.reshape(shape_1,shape_2, -1))#.view(-1, self.hidden_dim))
        #print("output_vb",output_vb.device)
        
        # output_vb = self.hid_to_out(reduce(
        #     torch.Tensor.add,
        #     [input_vb, solver_result_1.reshape(shape_1,shape_2, -1), solver_result_8.reshape(shape_1,shape_2, -1),
        #      solver_result_9.reshape(shape_1,shape_2, -1),solver_result_10.reshape(shape_1,shape_2, -1)],
        #     torch.zeros_like(input_vb)  # optionally set initial element to avoid changing `x`
        # ))
        #print("solver_result_1",solver_result_1.shape)
        #print("solver_result_8",solver_result_8.shape)
        #print("solver_result_9",solver_result_9.shape)
        #print("solver_result_10",solver_result_10.shape)
        #print("hid_to_out",self.hid_to_out)
        #print("hid_to_out",self.hid_to_out.gate_up_proj.weight.shape)
        combined_results =  solver_result_1 +input_vb_reshaped+solver_result_2+solver_result_3
        combined_results = mask_add*combined_results#torch.mul(mask_add,combined_results)
        output_vb = self.hid_to_out(combined_results.reshape(shape_1,shape_2, -1))

        #print("output_vb",output_vb.shape)
        #output_vb = self.out_fix(output_vb.flatten())
        output_vb = output_vb.reshape(shape_1,shape_2, -1)
        #self.input_vb = self.vis.heatmap(input_vb_reshaped.data[:80,:80].clone().cpu().detach().float().numpy().tolist(), env="daim_17080800", win=self.input_vb, opts=dict(title="input_vb")) #self.vis.heatmap(val, env=self.refs, win=self.win_head, opts=dict(title="write_head"))
        #print("output_vb",output_vb.shape)
        return output_vb #torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value).view(int(self.batch_size), int(self.batch_size), 64) #F.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value)).view(int(self.batch_size), int(self.batch_size), 64)
        #print("(1, int(self.batch_size), self.output_dim)",(1, int(self.batch_size), self.output_dim))
        # if not self.training:
        #     #with torch.no_grad():
        #         self.read_vec_vb = self.accessor.forward(input_vb)
        #         #self.accessor.visual()
        #         output_vb = self.hid_to_out(torch.cat((input_vb.view(-1, self.hidden_dim),
        #                                                self.read_vec_vb.view(-1, self.read_vec_dim)), 1))
        #         #output_vb = torch.cat((input_vb.view(-1, self.hidden_dim)), 1)
        #         #output_vb = self.hid_to_out(torch.cat((input_vb.view(-1, self.hidden_dim)), 1))
        #         return F.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value)).view(int(self.batch_size), int(self.batch_size), 64)
        # else:
        #     #print("1.5.1")
    
        #     self.read_vec_vb = self.accessor.forward(input_vb)
        #     #print("1.5.2")
        #     #output_vb = torch.zeros_like(self.hid_to_out(torch.cat((input_vb.view(-1, self.hidden_dim),
        #     #                                       self.read_vec_vb.view(-1, self.read_vec_dim)), 1)))
        #     #output_vb = torch.cat((input_vb.view(-1, self.hidden_dim)), 1)
        #     #output_vb = self.hid_to_out((input_vb.view(-1, self.hidden_dim)))

        #     output_vb = self.hid_to_out(torch.cat((input_vb.view(-1, self.hidden_dim),
        #                                                self.read_vec_vb.view(-1, self.read_vec_dim)), 1))

        #     with torch.no_grad():
        #         self.accessor.visual()
        #     #print("1.5.3")
        #     return F.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value)).view(int(self.batch_size), int(self.batch_size), 64)
            #            return F.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value)).view(768, int(self.batch_size/768), self.output_dim)

    def forward(self, input_vb):
        # NOTE: the operation order must be the following: control, access{write, read}, output

        # 1. first feed {input, read_vec_{t-1}} to controller
        hidden_vb = self.controller.forward(input_vb, self.read_vec_vb)
        # 2. then we write to memory_{t-1} to get memory_{t}; then read from memory_{t} to get read_vec_{t}
        self.read_vec_vb = self.accessor.forward(hidden_vb)
        # 3. finally we concat the output from the controller and the current read_vec_{t} to get the final output
        output_vb = self.hid_to_out(torch.cat((hidden_vb.view(-1, self.hidden_dim),
                                               self.read_vec_vb.view(-1, self.read_vec_dim)), 1))

        # we clip the output values here
        return F.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value)).view(48, self.batch_size, self.output_dim)
