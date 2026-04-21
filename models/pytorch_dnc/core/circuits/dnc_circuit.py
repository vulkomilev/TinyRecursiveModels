from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

from ...core.circuit import Circuit
from ...core.controllers.lstm_controller import LSTMController as Controller
from ...core.accessors.dynamic_accessor import DynamicAccessor as Accessor
from models.layers import rms_norm, LinearSwish, SwiGLUAsym

class DNCCircuit(Circuit):
    def __init__(self, args):
        super(DNCCircuit, self).__init__(args)

        # functional components
        self.controller = Controller(self.controller_params)
        self.accessor = Accessor(self.accessor_params)

        # build model
        #self.hid_to_out = nn.Linear(self.hidden_dim + self.read_vec_dim, self.output_dim)
        #self.hid_to_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.hid_to_out =         self.mlp = SwiGLUAsym(
                    #input_size=128,
                    input_size=64,
                    hidden_size=64,#self.output_dim,
                    expansion=4,
                    inter_min=256
                )

        self._reset()

    def _init_weights(self):
        pass
