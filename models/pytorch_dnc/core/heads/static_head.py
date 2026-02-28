from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ...core.head import Head
from ...utils.similarities import batch_cosine_sim

class StaticHead(Head):
    def __init__(self, args):
        super(StaticHead, self).__init__(args)

        # build model
        # for content focus
        self.hid_2_key   = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_beta  = nn.Linear(self.hidden_dim, self.num_heads * 1)
        # for location focus
        self.hid_2_gate  = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_shift = nn.Linear(self.hidden_dim, self.num_heads * self.num_allowed_shifts)
        self.hid_2_gamma = nn.Linear(self.hidden_dim, self.num_heads * 1)

    def _content_focus(self, memory_vb):
        """
        variables needed:
            key_vb:    [batch_size x num_heads x mem_wid]
                    -> similarity key vector, to compare to each row in memory
                    -> by cosine similarity
            beta_vb:   [batch_size x num_heads x 1]
                    -> NOTE: refer here: https://github.com/deepmind/dnc/issues/9
                    -> \in (1, +inf) after oneplus(); similarity key strength
                    -> amplify or attenuate the pecision of the focus
            memory_vb: [batch_size x mem_hei   x mem_wid]
        returns:
            wc_vb:     [batch_size x num_heads x mem_hei]
                    -> the attention weight by content focus
        """
       # print("self.key_vb",self.key_vb.shape)
        #exit(0)
        K_vb = batch_cosine_sim(self.key_vb, memory_vb)  # [batch_size x num_heads x mem_hei]
        self.wc_vb = K_vb * self.beta_vb.expand_as(K_vb) # [batch_size x num_heads x mem_hei]
        self.wc_vb = F.softmax(self.wc_vb.transpose(0, 2)).transpose(0, 2)

    def _shift(self, wg_vb, shift_vb):
        """
        variables needed:
            wg_vb:    [batch_size x num_heads x mem_hei]
            shift_vb: [batch_size x num_heads x num_allowed_shifts]
                   -> sum=1; the shift weight vector
        returns:
            ws_vb:    [batch_size x num_heads x mem_hei]
                   -> the attention weight by location focus
        """
        batch_size = wg_vb.size(0)
        input_dim = wg_vb.size(2); assert input_dim == self.mem_hei
        filter_dim = shift_vb.size(2); assert filter_dim == self.num_allowed_shifts

        ws_vb = None
        #print("1 head static _shift",batch_size,self.num_heads)
        for i in range(batch_size): # for each head in each batch, the kernel is different ... seems there's no other way by doing the loop here
           for j in range(self.num_heads):
               #print("2 head static _shift")
               ws_tmp_vb = F.conv1d(wg_vb[i][j].unsqueeze(0).unsqueeze(0).repeat(1, 1, 3),
                                    shift_vb[i][j].unsqueeze(0).unsqueeze(0).contiguous(),
                                    padding=filter_dim//2)[:, :, input_dim:(2*input_dim)]
               if ws_vb is None:
                   ws_vb = ws_tmp_vb
               else:
                   ws_vb = torch.cat((ws_vb, ws_tmp_vb), 0)
               #print("3 head static _shift")
        #print("4 head static _shift")
        ws_vb = ws_vb.view(-1, self.num_heads, self.mem_hei)
        #print("5 head static _shift")
        # B = batch_size
        # H = self.num_heads
        # L = input_dim
        # K = filter_dim

        # # (B, H, L) -> (B*H, L)
        # x = wg_vb.reshape(B * H, L)

        # # build length 3*L without per-(i,j) repeat
        # # (B*H, 3*L) -> (1, B*H, 3*L)
        # x3 = x.repeat(1, 3).unsqueeze(0).contiguous()

        # # (B, H, K) -> (B*H, 1, K)  (one kernel per group)
        # w = shift_vb.reshape(B * H, 1, K).contiguous()

        # # grouped conv: each channel gets its own kernel
        # y = F.conv1d(x3, w, padding=K // 2, groups=B * H)  # (1, B*H, 3*L)

        # # take the middle segment: (1, B*H, L)
        # y_mid = y[:, :, L:2 * L]

        # # match your old cat output shape: (B*H, 1, L)
        # ws_vb = y_mid.permute(1, 0, 2).contiguous()
        return ws_vb

    def _location_focus(self):
        """
        variables needed:
            wl_prev_vb: [batch_size x num_heads x mem_hei]
            wc_vb:      [batch_size x num_heads x mem_hei]
            gate_vb:    [batch_size x num_heads x 1]
                     -> \in (0, 1); the interpolation gate
            shift_vb:   [batch_size x num_heads x num_allowed_shifts]
                     -> sum=1; the shift weight vector
            gamma_vb:   [batch_size x num_heads x 1]
                     -> >=1; the sharpening vector
        returns:
            wl_curr_vb: [batch_size x num_heads x mem_hei]
                     -> the attention weight by location focus
        """
        #print("1 head static _location_focus")
        self.gate_vb = self.gate_vb.expand_as(self.wc_vb)
        #print("2 head static _location_focus")
        wg_vb = self.wc_vb * self.gate_vb + self.wl_prev_vb * (1. - self.gate_vb)
        #print("3 head static _location_focus")
        ws_vb = self._shift(wg_vb, self.shift_vb)
        #print("4 head static _location_focus")
        wp_vb = ws_vb.pow(self.gamma_vb.expand_as(ws_vb))
        # self.wl_curr_vb = wp_vb / wp_vb.sum(2).expand_as(wp_vb)               # 0.1.12
        #print("5 head static _location_focus")
        self.wl_curr_vb = wp_vb / wp_vb.sum(2, keepdim=True).expand_as(wp_vb)   # 0.2.0

    def forward(self, hidden_vb, memory_vb):
        # outputs for computing addressing for heads
        # NOTE: to be consistent w/ the dnc paper, we use
        # NOTE: sigmoid to constrain to [0, 1]
        # NOTE: oneplus to constrain to [1, +inf]
        # print("memory_vb",memory_vb.shape)
        #print("hidden_vb",hidden_vb.shape)
        #print("F.tanh(self.hid_2_key(hidden_vb))",F.tanh(self.hid_2_key(hidden_vb)).shape)
        #print("F.tanh(self.hid_2_key(hidden_vb))",F.tanh(self.hid_2_key(hidden_vb)).view(-1, self.num_heads, self.mem_wid).shape)

        #print("1 head static")
        self.key_vb   = F.tanh(self.hid_2_key(hidden_vb)).view(-1, self.num_heads, self.mem_wid)    # TODO: relu to bias the memory to store positive values ??? check again
        #print("2 head static")
        self.beta_vb  = F.softplus(self.hid_2_beta(hidden_vb)).view(-1, self.num_heads, 1)          # beta >=1: https://github.com/deepmind/dnc/issues/9
        #print("3 head static")
        self.gate_vb  = F.sigmoid(self.hid_2_gate(hidden_vb)).view(-1, self.num_heads, 1)           # gate /in (0, 1): interpolation gate, blend wl_{t-1} & wc
        #print("4 head static")
        self.shift_vb = F.softmax(self.hid_2_shift(hidden_vb).view(-1, self.num_heads, self.num_allowed_shifts).transpose(0, 2)).transpose(0, 2)    # shift: /sum=1
        #print("5 head static")
        self.gamma_vb = (1. + F.softplus(self.hid_2_gamma(hidden_vb))).view(-1, self.num_heads, 1)  # gamma >= 1: sharpen the final weights
        #print("6 head static")
        # now we compute the addressing mechanism
        self._content_focus(memory_vb)
        #print("7 head static")
        self._location_focus()
        #print("8 head static")