import torch
from torch import nn
from typing import List, Literal, Tuple
from constants import par
import numpy as np
import scipy

class RNN_stdp(torch.jit.ScriptModule):

    def __init__(self,
        n_input: int,
        n_context: int,
        n_output: int,
        hidden_dim: int,
        activation: Literal["tanh", "relu"] = "relu",
        alpha: float = 0.9,
        ) -> None:
        super(RNN_stdp, self).__init__()
        # Load the input activity, the target data, and the training mask for this batch of trials
        #self.input_data = tf.unstack(input_data, axis=0)
        #self.target_data = target_data
        #self.mask = mask
        print('n_input' , n_input)
        print('n_context',n_context)
        print('n_output',n_output)
        self.device="cuda:0"
        self.n_=n_input
        self.h_=hidden_dim
        self.input = nn.Linear(n_input, hidden_dim).to(self.device)
        self.context = nn.Linear(n_context, hidden_dim).to(self.device)
        self.output = nn.Linear(hidden_dim, n_output).to(self.device)

        self.initialize_weights()
        print('self.var_dicth]',self.var_dict['h'].shape)
        self.h = torch.Tensor([]).to(self.device)
        self.syn_x = torch.Tensor([]).to(self.device)
        self.syn_u =torch.Tensor([]).to(self.device)
        self.y = torch.Tensor([]).to(self.device)
        
        #self.h = []
        #self.syn_x = []
        #self.syn_u = []
        #self.y = []
        
        self.alpha_std = torch.tensor(par['alpha_std']).to(self.device)
        self.alpha_stf = torch.tensor(par['alpha_stf']).to(self.device)
        self.dt_sec = par['dt_sec'] 
        self.dynamic_synapse =  torch.tensor(par['dynamic_synapse']).to(self.device)
        self.U = torch.tensor(par['U']).to(self.device)
        self.act = nn.ReLU().to(self.device)
        self.alpha_neuron = par['alpha_neuron']
        self.noise_rnn = par['noise_rnn']
        self.b_rnn = self.var_dict['b_rnn'].to(self.device)
        
        #alpha_std = np.ones((1, par['n_hidden']), dtype=np.float32) #torch.Tensor(par['alpha_std'])
        #alpha_stf = np.ones((1, par['n_hidden']), dtype=np.float32) #par['alpha_stf']
        #dt_sec = 10/1000. #par['dt_sec'] 
        #dynamic_synapse = np.zeros((1, par['n_hidden']), dtype=np.float32) # par['dynamic_synapse']
        #U = np.ones((1, par['n_hidden']), dtype=np.float32) #par['U']

    def initialize_matrix(self,dimensions=(), connection_prob=.2, shape=0.1, scale=1.0 ):
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
                self.var_dict[name] = torch.tensor(par[k],requires_grad=True ).to(self.device) 
                if name == 'w_in':
                    dim_list= [self.n_, self.h_]
                    conn_prob=par['connection_prob']/par['num_receptive_fields']
                    w_in = self.initialize_matrix(dim_list, conn_prob)
                    self.var_dict[name] =torch.tensor(w_in,requires_grad=True )  
                #self.var_dict[name].
        #torch.autograd.Variable(torch.Tensor(par[k]))

        self.syn_x_init = torch.Tensor(par['syn_x_init']).to(self.device)
        self.syn_u_init = torch.Tensor(par['syn_u_init']).to(self.device)
        if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            act = nn.ReLU()
            out = act(self.var_dict['w_rnn'])
            self.w_rnn = torch.Tensor(par['EI_matrix']).to(self.device) @ out.to(self.device)
        else:
            self.w_rnn  = self.var_dict['w_rnn'].to(self.device)

    def rnn_cell(self, rnn_input, h, syn_x, syn_u): #, alpha_std,alpha_stf,dt_sec,dynamic_synapse, U):
        # Update neural activity and short-term synaptic plasticity values
        # Update the synaptic plasticity paramaters
        # implement both synaptic short term facilitation and depression
        syn_x += (self.alpha_std*(1-syn_x) - self.dt_sec*syn_u*syn_x*h)*self.dynamic_synapse
        syn_u += (self.alpha_stf*(self.U-syn_u) + self.dt_sec*self.U*(1-syn_u)*h)*self.dynamic_synapse

        syn_x = torch.min(torch.tensor([1.]).to(self.device), self.act(syn_x.to(self.device)))
        syn_u = torch.min(torch.tensor([1.]).to(self.device), self.act(syn_u.to(self.device)))
        h_post = syn_u*syn_x*h

        #else:
        #    # no synaptic plasticity
        #    h_post = h

        # Update the hidden state. Only use excitatory projections from input layer to RNN
        # All input and RNN activity will be non-negative
        print('rnn_input.sahpe',rnn_input.shape)
        print('self.var_dict[w_in].shape',self.var_dict['w_in'].shape)
        print('hpost',h_post.shape)
        print('self.w_rnn.sape',self.w_rnn.shape)
        print('self.var_dict[b_rnn].shape',self.var_dict['b_rnn'].shape)
        print('h.shape',h.shape)
        print('self.noise_rnn',self.noise_rnn)
        frac_activ= h * (1-self.alpha_neuron)
        rnn_1 = rnn_input.to("cuda:0") @ self.act(self.var_dict['w_in'].to("cuda:0"))
        rnn_2 = h_post @ self.w_rnn 
        rnn_3 = self.var_dict['b_rnn']

        rnn_ = self.alpha_neuron * ( rnn_1.to(self.device)
            + rnn_2.to("cuda:0") + rnn_3.to(self.device)) 
        norm_inp = torch.normal( torch.zeros(h.shape), self.noise_rnn)
        rnn_.to("cuda:0")
        norm_inp.to("cuda:0")
        frac_activ.to("cuda:0")
        inputs_fac = rnn_.to("cuda:0") + norm_inp.to(self.device)
        inputs_fac.to("cuda:0")
        h = self.act(frac_activ.to("cuda:0")  + inputs_fac.to("cuda:0")) #, dtype=torch.float32))

        return h, syn_x, syn_u
    #def run_model(self):
    @torch.jit.script_method
    def forward(
        self,
        stim_input: torch.Tensor,
        context: torch.Tensor,
    )-> Tuple[torch.Tensor, torch.Tensor]: #]:
        # Main model loop
        
        inputs = stim_input.unbind(1)

        print('HELLLOOO')
        h = self.var_dict['h'] # torch.tensor(self.var_dict['h'])
        syn_x = self.syn_x_init
        syn_u = self.syn_u_init
                
        
        # Loop through the neural inputs to the RNN, indexed in time
        #for rnn_input in  inputs:
        for rnn_input in inputs:
            print('HELLO')
            #print(rnn_input.shape)
            print('syn_x.shape',syn_x.shape)
            print('h shape',h.shape)
            h, syn_x, syn_u = self.rnn_cell(rnn_input, h, syn_x, syn_u) 
            print('h.shape',h.shape)
            self.h = torch.cat((self.h,h))
            self.syn_x = torch.cat((self.syn_x,syn_x))
            self.syn_u = torch.cat((self.syn_u,syn_u))
            self.y = torch.cat((self.y,h @ self.act(self.var_dict['w_out']) + self.var_dict['b_out']))

        #self.h = torch.stack(self.h)
        #self.syn_x = torch.stack(self.syn_x)
        #self.syn_u = torch.stack(self.syn_u)
        #output
        #self.y = torch.stack(self.y)
        
        print('self.h.shape',self.h.shape)
        print('self.y.shape',self.y.shape)
        outputs = self.y
        activity = self.h
 
        return outputs, activity #, class_preds, dim=1)


class RNN(torch.jit.ScriptModule):

    def __init__(
        self,
        n_input: int,
        n_context: int,
        n_output: int,
        hidden_dim: int,
        activation: Literal["tanh", "relu"] = "relu",
        alpha: float = 0.9,
    ) -> None:
        super(RNN, self).__init__()
        self.input = nn.Linear(n_input, hidden_dim)
        self.context = nn.Linear(n_context, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_output)

        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.w_rec = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.b_rec = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.xavier_uniform_(self.w_rec, gain=0.1)

        self.w_mask = torch.ones((hidden_dim, hidden_dim)) - torch.eye(hidden_dim)
        self.alpha = alpha if activation == "relu" else 0.0
        self.activation = activation
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()

        self.classifiers = Classifers(n_input=hidden_dim, n_classifiers=3)

    @torch.jit.script_method
    def forward(
        self,
        stim_input: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        inputs = stim_input.unbind(1)
        ctx = context.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        activity = torch.jit.annotate(List[torch.Tensor], [])
        class_preds = torch.jit.annotate(List[torch.Tensor], [])

        h = self.h0
        w_rec = self.w_mask.to(device=self.w_rec.device) * self.w_rec if self.activation == "relu" else self.w_rec

        for i in range(len(inputs)):
            h = self.alpha * h + torch.mm(h, w_rec) + self.b_rec + self.input(inputs[i]) + self.context(ctx[i])
            h = self.act(h)
            h = torch.clip(h, -1.0, 10.0)
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


