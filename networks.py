import attr
import copy
import torch
from torch import nn
import numpy as np
from typing import List, Literal, Tuple

#@attr.s(auto_attribs=True)
"""
class BaseModel(torch.jit.ScriptModule):
    n_input: int = attr.ib(default=36)
    n_context: int = attr.ib(default=200)
    n_output: int = attr.ib(default=9)
    n_hidden: int = attr.ib(default=500)
    alpha: float = attr.ib(default=0.8)
    tau_neuron: float = attr.ib(default=100.0)
    tau_slow: float = attr.ib(default=1000.0)
    tau_fast: float = attr.ib(default=200.0)
    dt: float = attr.ib(default=20.0)
    exc_fraction: float = attr.ib(default=0.8)
    noise_std: float = attr.ib(default=0.01)

    alpha_neuron: float = attr.ib(init=False)
    alpha_x: float = attr.ib(init=False)
    alpha_u: float = attr.ib(init=False)
    exc_inh: torch.Tensor = attr.ib(init=False)
"""
class BaseModel(torch.jit.ScriptModule):
    def __init__(self, n_input=36,n_context=200,n_output=9,n_hidden=500,alpha=.8,tau_neuron=100,tau_slow=1000,tau_fast=200,dt=20,exc_fraction=.8,noise_std=.01):
        super(BaseModel, self).__init__()
        self.n_input=n_input
        self.n_context=n_context
        self.n_output=n_output
        self.n_hidden = n_hidden
        self.alpha=alpha
        self.tau_neuron = tau_neuron 
        self.tau_slow = tau_slow 
        self.tau_fast=tau_fast
        self.dt=dt 
        self.exc_fraction = exc_fraction 
        self.noise_std=noise_std

        
        #self.exc_inh = False
        #def __attrs_post_init__(self):
        """ Initialize various parameters
        STDP parameters
            x: available neurotransmitter
            u: residual calcium
            even number neurons: depressing
            odd number neurons: facilitating
        """
        self.dt_sec = self.dt / 1000.0
        self.alpha_neuron = self.dt / self.tau_neuron
        self.alpha_x = torch.ones(self.n_hidden) * self.dt / self.tau_slow
        self.alpha_x[1::2] = torch.ones(self.n_hidden)[1::2] * self.dt / self.tau_fast
        self.alpha_u = torch.ones(self.n_hidden) * self.dt / self.tau_slow
        self.alpha_u[1::2] = torch.ones(self.n_hidden)[1::2] * self.dt / self.tau_fast
        self.U = 0.45 * torch.ones(self.n_hidden)
        self.U[1::2] = 0.15

        n_exc = int(self.n_hidden) * self.exc_fraction
        self.exc_inh = torch.eye(self.n_hidden).to(self.device)
        self.exc_inh[int(n_exc):] * -1.0

#@attr.s(auto_attribs=True)
class RNN(BaseModel):
    def __init__(self,n_input=36,n_context=200,n_output=9,n_hidden=500,alpha=.8,tau_neuron=100,tau_slow=1000,tau_fast=200,dt=20,exc_fraction=.8,noise_std=.01):
        super(RNN, self).__init__()
        self.h_init = nn.Parameter(torch.zeros(1, self.n_hidden))
        
        self.device=self.h_init.device() #'cuda:0'
        self.W = torch.ones(size=(self.n_hidden, self.n_hidden))
        w = np.random.gamma(shape=0.1, scale=1.0, size=(self.n_hidden, self.n_hidden))
        self.w_rec = nn.Parameter(data=torch.from_numpy(w))
        self.b_rec = nn.Parameter(torch.zeros(1, self.n_hidden))

        self.w_mask = torch.ones((self.n_hidden, self.n_hidden)) - torch.eye(self.n_hidden)
        self.w_mask.to(device=self.device)
        self.classifiers = Classifers(n_input=self.n_hidden, n_classifiers=3)
        self.relu = nn.ReLU()
        
        self.input = nn.Linear(self.n_input, self.n_hidden, bias=False)
        self.context = nn.Linear(self.n_context, self.n_hidden, bias=False)
        self.output = nn.Linear(self.n_hidden, self.n_output)
        
        self.b_rnn = torch.zeros((1,self.n_hidden))#.to(self.device)
        
    def _init_before_trial(self):
        """Initialize EXC and INH weights, and STDP initial values"""
        print('self.exc_inh.device',self.exc_inh.device)
        print('self.w_mask.device',self.w_mask.device)
        self.W = (self.exc_inh @ self.relu(self.w_rec.float().to(device=self.device))) * self.w_mask.to(device=self.device)
        syn_x = torch.ones(self.n_hidden)
        syn_u = self.U.float() * torch.ones(self.n_hidden)
        #h = copy.copy(self.h_init)
        #h = copy.deepcopy(self.h_init)

        return self.h_init, syn_x, syn_u
    
    def rnn_cell(self, stim, context, h, syn_x, syn_u):
        """Update neural activity and short-term synaptic plasticity values"""

        # implement both synaptic short term facilitation and depression
        syn_x += (self.alpha_x * (1 - syn_x) - self.dt_sec * syn_u * syn_x * h)
        syn_u += (self.alpha_x * (self.U - syn_u) + self.dt_sec * self.U * (1 - syn_u) * h)
        syn_x = torch.clip(syn_x, 0.0, 1.0)
        syn_u = torch.clip(syn_u, 0.0, 1.0)
        h_post = syn_u * syn_x * h

        # Update the hidden state.
        stim_input = self.input(stim)
        context_input = self.context(context)
        h = h * (1 - self.alpha) + self.alpha * (
            stim_input + context_input + h_post @ self.W + self.b_rnn
        ) + self.noise_std * torch.randn(h.size()).to(device=h.device)
        h = self.relu(h)

        return h, syn_x, syn_u

    @torch.jit.script_method
    def forward(
        self,
        stim_input: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        h, syn_x, syn_u = self._init_before_trial()

        inputs = stim_input.unbind(1)
        ctx = context.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        activity = torch.jit.annotate(List[torch.Tensor], [])
        class_preds = torch.jit.annotate(List[torch.Tensor], [])

        for i in range(len(inputs)):
            h, syn_x, syn_u = self.rnn_cell(inputs[i], ctx[i], h, syn_x, syn_u)
            out = self.output(h)
            outputs += [out]
            activity += [h]

            p = self.classifiers(h)
            class_preds += [torch.permute(p, (1, 0, 2))]

        return torch.stack(outputs, dim=1), torch.stack(activity, dim=1), torch.stack(class_preds, dim=1)


class LSTM_XX(torch.jit.ScriptModule):

    def __init__(
        self,
        n_input: int,
        n_context: int,
        n_output: int,
        n_first_batch: int,
        hidden_dim: int,
    ):

        super(LSTM, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_dim, n_input))
        self.weight_ch = nn.Parameter(torch.randn(4 * hidden_dim, n_first_batch))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_dim))
        self.output = nn.Linear(hidden_dim, n_output)
        self.value = nn.Linear(hidden_dim, 1)

        self.context = nn.Linear(n_context - n_first_batch, n_first_batch)

        nn.init.xavier_uniform_(self.weight_ih, gain=1.0)
        #nn.init.xavier_uniform_(self.context, gain=1.0)
        nn.init.xavier_uniform_(self.weight_hh, gain=1.0)

        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.hidden_dim = hidden_dim
        self.n_first_batch = n_first_batch

        self.classifiers = Classifers(n_input=hidden_dim, n_classifiers=3)

    @torch.jit.script_method
    def forward(
        self,
        stim_input: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        inputs = stim_input.unbind(1)
        ctx = context.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        pred_values = torch.jit.annotate(List[torch.Tensor], [])
        class_preds = torch.jit.annotate(List[torch.Tensor], [])
        hx: torch.Tensor = self.h0
        cx: torch.Tensor = self.c0

        for i in range(len(inputs)):

            ctx0, ctx1 = ctx[i][:, :self.n_first_batch], ctx[i][:, self.n_first_batch:]
            ctx0 = ctx0 + self.context(ctx1)

            gates = (torch.mm(inputs[i], self.weight_ih.t()) + self.bias_ih +
                     torch.mm(hx, self.weight_hh.t()) + self.bias_hh +
                     torch.mm(ctx0, self.weight_ch.t()))

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            cx = (forgetgate * cx) + (ingate * cellgate)
            hx = outgate * torch.tanh(cx)
            out = self.output(hx)
            val = self.value(hx)
            outputs += [out]
            pred_values += [val]

            p = self.classifiers(h)
            class_preds += [torch.permute(p, (1, 0, 2))]

        return (torch.stack(outputs, dim=1), torch.stack(pred_values, dim=1)), torch.stack(class_preds, dim=1)


class LSTM(torch.jit.ScriptModule):

    def __init__(
        self,
        n_input: int,
        n_context: int,
        n_output: int,
        hidden_dim: int,
    ):

        super(LSTM, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_dim, n_input))
        self.context = nn.Parameter(torch.randn(4 * hidden_dim, n_context))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_dim))
        self.output = nn.Linear(hidden_dim, n_output)
        self.value = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.weight_ih, gain=1.0)
        nn.init.xavier_uniform_(self.context, gain=1.0)
        nn.init.xavier_uniform_(self.weight_hh, gain=1.0)

        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.hidden_dim = hidden_dim

        self.classifiers = Classifers(n_input=2*hidden_dim, n_classifiers=3)

    @torch.jit.script_method
    def forward(
        self,
        stim_input: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        inputs = stim_input.unbind(1)
        ctx = context.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        pred_values = torch.jit.annotate(List[torch.Tensor], [])
        class_preds = torch.jit.annotate(List[torch.Tensor], [])
        hx: torch.Tensor = self.h0
        cx: torch.Tensor = self.c0

        # context_weight = 10 * self.context / torch.sqrt(torch.sum(self.context ** 2, dim=0, keepdim=True))
        context_weight = self.context
        for i in range(len(inputs)):

            gates = (torch.mm(inputs[i], self.weight_ih.t()) + self.bias_ih +
                     torch.mm(hx, self.weight_hh.t()) + self.bias_hh +
                     torch.mm(ctx[i], context_weight.t()))

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            cx = (forgetgate * cx) + (ingate * cellgate)
            hx = outgate * torch.tanh(cx)
            out = self.output(hx)
            val = self.value(hx)
            outputs += [out]
            pred_values += [val]

            p = self.classifiers(torch.cat((hx.detach(), cx.detach()), dim=-1))
            class_preds += [torch.permute(p, (1, 0, 2))]

        return torch.stack(outputs, dim=1), torch.stack(pred_values, dim=1), torch.stack(class_preds, dim=1)


class LSTM_ctx_bottleneck(torch.jit.ScriptModule):

    def __init__(
        self,
        n_input: int,
        n_context: int,
        n_output: int,
        hidden_dim: int,
        ctx_hidden_dim: int = 128,
    ):
        # context_hidden_layer will add a bottleneck layer between the rule tuned
        # neurons and the LSTM
        super(LSTM_ctx_bottleneck, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_dim, n_input))
        self.weight_ch = nn.Parameter(torch.randn(4 * hidden_dim, ctx_hidden_dim))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_dim))

        self.output = nn.Linear(hidden_dim, n_output)
        self.value = nn.Linear(hidden_dim, 1)
        self.context = nn.Linear(n_context, ctx_hidden_dim)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.weight_ih, gain=1.0)
        nn.init.xavier_uniform_(self.weight_ch, gain=1.0)
        nn.init.xavier_uniform_(self.weight_hh, gain=1.0)

        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.hidden_dim = hidden_dim

        self.classifiers = Classifers(n_input=2 * hidden_dim, n_classifiers=3)

    @torch.jit.script_method
    def forward(
        self,
        stim_input: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        inputs = stim_input.unbind(1)
        ctx = context.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        pred_values = torch.jit.annotate(List[torch.Tensor], [])
        class_preds = torch.jit.annotate(List[torch.Tensor], [])
        hx: torch.Tensor = self.h0
        cx: torch.Tensor = self.c0

        for i in range(len(inputs)):

            ctx_hidden = self.context(ctx[i])
            # ctx_hidden = self.relu(ctx_hidden)   # Removing this seems to help
            gates = (torch.mm(inputs[i], self.weight_ih.t()) + self.bias_ih +
                     torch.mm(hx, self.weight_hh.t()) + self.bias_hh +
                     torch.mm(ctx_hidden, self.weight_ch.t()))

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            cx = (forgetgate * cx) + (ingate * cellgate)
            hx = outgate * torch.tanh(cx)
            out = self.output(hx)
            val = self.value(hx)
            outputs += [out]
            pred_values += [val]

            p = self.classifiers(torch.cat((hx.detach(), cx.detach()), dim=-1))
            class_preds += [torch.permute(p, (1, 0, 2))]

        return torch.stack(outputs, dim=1), torch.stack(pred_values, dim=1), torch.stack(class_preds, dim=1)


class RNN_stdp(torch.jit.ScriptModule):

    def __init__(self,
                 n_input: int,
                 n_context: int,
                 n_output: int,
                 ) -> None:
        super(RNN_stdp, self).__init__()
        # Load the input activity, the target data, and the training mask for this batch of trials
        # self.input_data = tf.unstack(input_data, axis=0)
        # self.target_data = target_data
        # self.mask = mask
        print('n_input', n_input)
        print('n_context', n_context)
        print('n_output', n_output)
        self.device = "cuda:0"
        self.n_ = n_input
        self.h_ = hidden_dim

        self.input = nn.Linear(n_input, hidden_dim).to(self.device)
        self.context = nn.Linear(n_context, hidden_dim).to(self.device)
        self.context = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.output = nn.Linear(hidden_dim, n_output).to(self.device)

        self.initialize_weights()
        print('self.var_dicth]', self.var_dict['h'].shape)
        self.h = torch.Tensor([]).to(self.device)
        self.syn_x = torch.Tensor([]).to(self.device)
        self.syn_u = torch.Tensor([]).to(self.device)
        self.y = torch.Tensor([]).to(self.device)

        # self.h = []
        # self.syn_x = []
        # self.syn_u = []
        # self.y = []

        self.alpha_std = torch.tensor(par['alpha_std']).to(self.device)
        self.alpha_stf = torch.tensor(par['alpha_stf']).to(self.device)
        self.dt_sec = par['dt_sec']
        self.dynamic_synapse = torch.tensor(par['dynamic_synapse']).to(self.device)
        self.U = torch.tensor(par['U']).to(self.device)
        self.act = nn.ReLU().to(self.device)
        self.alpha_neuron = par['alpha_neuron']
        self.noise_rnn = par['noise_rnn']
        self.b_rnn = self.var_dict['b_rnn'].to(self.device)

        # alpha_std = np.ones((1, par['n_hidden']), dtype=np.float32) #torch.Tensor(par['alpha_std'])
        # alpha_stf = np.ones((1, par['n_hidden']), dtype=np.float32) #par['alpha_stf']
        # dt_sec = 10/1000. #par['dt_sec']
        # dynamic_synapse = np.zeros((1, par['n_hidden']), dtype=np.float32) # par['dynamic_synapse']
        # U = np.ones((1, par['n_hidden']), dtype=np.float32) #par['U']

    def initialize_matrix(self, dimensions=(), connection_prob=.2, shape=0.1, scale=1.0):
        w = np.random.gamma(shape, scale, size=dimensions)
        w *= (np.random.rand(*dimensions) < connection_prob)

        return np.float32(w)

    def initialize_weights(self):
        # Initialize all weights. biases, and initial values

        self.var_dict = {}
        # all keys in par with a suffix of '0' are initial values of trainable variables
        for k, v in par.items():
            if k[-1] == '0':
                name = k[:-1]
                print('name', name)
                print(par[k].shape)
                self.var_dict[name] = torch.tensor(par[k], requires_grad=True).to(self.device)
                if name == 'w_in':
                    dim_list = [self.n_, self.h_]
                    conn_prob = par['connection_prob'] / par['num_receptive_fields']
                    w_in = self.initialize_matrix(dim_list, conn_prob)
                    self.var_dict[name] = torch.tensor(w_in, requires_grad=True)
                    # self.var_dict[name].
        # torch.autograd.Variable(torch.Tensor(par[k]))

        self.syn_x_init = torch.Tensor(par['syn_x_init']).to(self.device)
        self.syn_u_init = torch.Tensor(par['syn_u_init']).to(self.device)
        if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            act = nn.ReLU()
            out = act(self.var_dict['w_rnn'])
            self.w_rnn = torch.Tensor(par['EI_matrix']).to(self.device) @ out.to(self.device)
        else:
            self.w_rnn = self.var_dict['w_rnn'].to(self.device)

    def rnn_cell(self, rnn_input, h, syn_x, syn_u):  # , alpha_std,alpha_stf,dt_sec,dynamic_synapse, U):
        # Update neural activity and short-term synaptic plasticity values
        # Update the synaptic plasticity paramaters
        # implement both synaptic short term facilitation and depression
        syn_x += (self.alpha_std * (1 - syn_x) - self.dt_sec * syn_u * syn_x * h) * self.dynamic_synapse
        syn_u += (self.alpha_stf * (self.U - syn_u) + self.dt_sec * self.U * (1 - syn_u) * h) * self.dynamic_synapse

        syn_x = torch.min(torch.tensor([1.]).to(self.device), self.act(syn_x.to(self.device)))
        syn_u = torch.min(torch.tensor([1.]).to(self.device), self.act(syn_u.to(self.device)))
        h_post = syn_u * syn_x * h

        # else:
        #    # no synaptic plasticity
        #    h_post = h

        # Update the hidden state. Only use excitatory projections from input layer to RNN
        # All input and RNN activity will be non-negative
        print('rnn_input.sahpe', rnn_input.shape)
        print('self.var_dict[w_in].shape', self.var_dict['w_in'].shape)
        print('hpost', h_post.shape)
        print('self.w_rnn.sape', self.w_rnn.shape)
        print('self.var_dict[b_rnn].shape', self.var_dict['b_rnn'].shape)
        print('h.shape', h.shape)
        print('self.noise_rnn', self.noise_rnn)
        frac_activ = h * (1 - self.alpha_neuron)
        rnn_1 = rnn_input.to("cuda:0") @ self.act(self.var_dict['w_in'].to("cuda:0"))
        rnn_2 = h_post @ self.w_rnn
        rnn_3 = self.var_dict['b_rnn']

        rnn_ = self.alpha_neuron * (rnn_1.to(self.device)
                                    + rnn_2.to("cuda:0") + rnn_3.to(self.device))
        norm_inp = torch.normal(torch.zeros(h.shape), self.noise_rnn)
        rnn_.to("cuda:0")
        norm_inp.to("cuda:0")
        frac_activ.to("cuda:0")
        inputs_fac = rnn_.to("cuda:0") + norm_inp.to(self.device)
        inputs_fac.to("cuda:0")
        h = self.act(frac_activ.to("cuda:0") + inputs_fac.to("cuda:0"))  # , dtype=torch.float32))

        return h, syn_x, syn_u

    # def run_model(self):
    @torch.jit.script_method
    def forward(
            self,
            stim_input: torch.Tensor,
            context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # ]:
        # Main model loop

        inputs = stim_input.unbind(1)

        print('HELLLOOO')
        h = self.var_dict['h']  # torch.tensor(self.var_dict['h'])
        syn_x = self.syn_x_init
        syn_u = self.syn_u_init

        # Loop through the neural inputs to the RNN, indexed in time
        # for rnn_input in  inputs:
        for rnn_input in inputs:
            print('HELLO')
            # print(rnn_input.shape)
            print('syn_x.shape', syn_x.shape)
            print('h shape', h.shape)
            h, syn_x, syn_u = self.rnn_cell(rnn_input, h, syn_x, syn_u)
            print('h.shape', h.shape)
            self.h = torch.cat((self.h, h))
            self.syn_x = torch.cat((self.syn_x, syn_x))
            self.syn_u = torch.cat((self.syn_u, syn_u))
            self.y = torch.cat((self.y, h @ self.act(self.var_dict['w_out']) + self.var_dict['b_out']))

        # self.h = torch.stack(self.h)
        # self.syn_x = torch.stack(self.syn_x)
        # self.syn_u = torch.stack(self.syn_u)
        # output
        # self.y = torch.stack(self.y)

        print('self.h.shape', self.h.shape)
        print('self.y.shape', self.y.shape)
        outputs = self.y
        activity = self.h

        return outputs, activity  # , class_preds, dim=1)


class Classifers(nn.Module):
    def __init__(
            self,
            n_input: int,
            n_classifiers: int,
            n_output: int = 8,
            bottleneck_dim: int = 2,
    ):
        super(Classifers, self).__init__()
        self.linear = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_input, bottleneck_dim),
                    nn.Linear(bottleneck_dim, n_output),
                )
                for _ in range(n_classifiers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [f(x.detach()) for f in self.linear]
        return torch.stack(y)


