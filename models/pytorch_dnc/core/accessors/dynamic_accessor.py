from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

from ...core.symbolic_logic import SymbolicLogic
from ...core.accessor import Accessor
from ...core.heads.dynamic_write_head import DynamicWriteHead as WriteHead
from ...core.heads.dynamic_read_head import DynamicReadHead as ReadHead
from ...core.memory import External2DMemory as ExternalMemory
import visdom

class DynamicAccessor(Accessor):
    def __init__(self, args):
        super(DynamicAccessor, self).__init__(args)
        # logging
        self.logger = args.logger
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        # dynamic-accessor-specific params
        self.read_head_params.num_read_modes = self.write_head_params.num_heads * 2 + 1

        self.logger.warning("<--------------------------------===> Accessor:   {WriteHead, ReadHead, Memory}")

        # functional components
        self.usage_vb = None    # for dynamic allocation, init in _reset
        self.link_vb = None     # for temporal link, init in _reset
        self.preced_vb = None   # for temporal link, init in _reset
        self.write_heads = WriteHead(self.write_head_params)
        self.read_heads = ReadHead(self.read_head_params)
        self.symbolic_logic = SymbolicLogic() 
        self.memory = ExternalMemory(self.memory_params)

        self._reset()

    def reset_visual(self):
        self.vis = visdom.Visdom()
        self.write_heads._reset_visual()
        self.read_heads._reset_visual()
        self.memory._reset_visual()

    def _init_weights(self):
        pass

    def _reset_states(self):
        # reset the usage (for dynamic allocation) & link (for temporal link)
        self.usage_vb  = Variable(self.usage_ts).type(self.dtype)
        self.link_vb   = Variable(self.link_ts).type(self.dtype)
        self.preced_vb = Variable(self.preced_ts).type(self.dtype)
        # we reset the write/read weights of heads
        self.write_heads._reset_states()
        self.read_heads._reset_states()
        # we also reset the memory to bias value
        self.memory._reset_states()

    def _reset(self):           # NOTE: should be called at __init__
        self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        # reset internal states
        self.vis = visdom.Visdom()
        self.usage_ts  = torch.zeros(self.batch_size, self.mem_hei)
        self.link_ts   = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei, self.mem_hei)
        self.preced_ts = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei)
        self._reset_states()
    def visual(self):
        super().visual()
        #if self.visualize:
            #self.win_head = "win_write_head"
       #print("self.hidden_vb.data[0].shape",self.hidden_vb.shape)
        if len(self.hidden_vb.shape) == 1:
            self.show_hidden_vb = self.hidden_vb.expand(1, -1)
        else:
            self.show_hidden_vb = self.hidden_vb
        if not self.training:
          print('1111')
          self.win_head = self.vis.heatmap(self.show_hidden_vb.data.clone().cpu().transpose(0, 1).float().numpy().tolist(), env=self.refs, win=self.win_head, opts=dict(title="hidden_vb")) #self.vis.heatmap(val, env=self.refs, win=self.win_head, opts=dict(title="write_head"))

    def _symbolic_processing(self,memory):
        for b_n in range(len(memory)):
            a = memory[b_n][0]
            b = memory[b_n][1]
            memory[b_n][3] = a + b
        return memory

    def forward(self, hidden_vb):
        # 1. first we update the usage using the read/write weights from {t-1}
        #print("1.5.1.1")
        #self.memory._reset()
        self.hidden_vb = hidden_vb
        self.usage_vb = self.write_heads._update_usage(self.usage_vb)

        #print("1.5.1.2")
        self.usage_vb = self.read_heads._update_usage(hidden_vb, self.usage_vb)

        self.memory.memory_vb = self.symbolic_logic.forward(self.memory.memory_vb)
        
        #print("1.5.1.3")
        # 2. then write to memory_{t-1} to get memory_{t}
        self.memory.memory_vb = self.write_heads.forward(hidden_vb, self.memory.memory_vb, self.usage_vb)
        #print("1.5.1.4")
        #my idea memory symbolic intervetion
        #self.memory.memory_vb = self._symbolic_processing(self.memory.memory_vb)
        # 3. then we update the temporal link
        self.link_vb, self.preced_vb = self.write_heads._temporal_link(self.link_vb, self.preced_vb)
        #print("1.5.1.5")
        # 4. then read from memory_{t} to get read_vec_{t}
        read_vec_vb = self.read_heads.forward(hidden_vb, self.memory.memory_vb, self.link_vb, self.write_head_params.num_heads)
        #print("1.5.1.6")
        return read_vec_vb
