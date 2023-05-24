import copy
import torch
from torch import nn
import numpy as np
from typing import List, Literal, Tuple

class RNN(nn.Module):

    def __init__(self, **kwargs):
        super(RNN, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._initialize_dependents()
        self._initialize_weights()

    def reset_context_weights(self):
        nn.init.xavier_uniform_(self.context.weight, gain=1.0)

    def _initialize_weights(self):

        # Initialize recurrent weights, multiply EXC->INH and INH->EXC by 2.0
        w = np.random.gamma(shape=0.2, scale=1.0, size=(self.n_hidden, self.n_hidden)).astype(np.float32)
        w /= 10.0
        w[self.n_exc:] *= 2.0
        w[:, self.n_exc:] *= 2.0
        self.w_rec = nn.Parameter(data=torch.from_numpy(w))

        # mask to prevent self-connections (autotapses)
        self.w_mask = torch.ones((self.n_hidden, self.n_hidden)) - torch.eye(self.n_hidden)

        # initial activity
        self.h_init = 0.1 * nn.Parameter(torch.zeros(1, self.n_hidden))

        # recurrent unit bias
        self.b_rec = nn.Parameter(torch.zeros(1, self.n_hidden))

        # weights and activations
        self.relu = nn.ReLU()
        self.input = nn.Linear(self.n_stimulus, self.n_hidden, bias=False)
        self.context = nn.Linear(self.n_context, self.n_hidden, bias=False)
        self.output = nn.Linear(self.n_hidden, self.n_output)

        # predict task variables from network activity, ot used to train network
        self.classifiers = Classifers(n_input=self.n_hidden, n_classifiers=3)

    def _initialize_dependents(self):

        self.dt_sec = self.dt / 1000.0
        self.alpha_neuron = self.dt / self.tau_neuron
        self.alpha_x = torch.ones(size=(self.n_hidden,)) * self.dt / self.tau_slow
        self.alpha_x[1::2] = torch.ones(size=(self.n_hidden,))[1::2] * self.dt / self.tau_fast
        self.alpha_u = torch.ones(size=(self.n_hidden,)) * self.dt / self.tau_slow
        self.alpha_u[1::2] = torch.ones(size=(self.n_hidden,))[1::2] * self.dt / self.tau_fast
        self.U = 0.45 * torch.ones(size=(self.n_hidden,))
        self.U[1::2] = 0.15

        self.n_exc = int(self.n_hidden * self.exc_fraction)
        self.exc_inh = torch.eye(self.n_hidden, dtype=torch.float32)
        self.exc_inh[self.n_exc:] *= -1.0
        
    def _init_before_trial(self):
        """Initialize EXC and INH weights, and STDP initial values"""
        device = self.w_rec.device
        self.U = self.U.to(device)
        self.alpha_x = self.alpha_x.to(device)
        self.alpha_u = self.alpha_u.to(device)
        self.W = (self.exc_inh.to(device) @ self.relu(self.w_rec)) * self.w_mask.to(device)
        syn_x = torch.ones(self.batch_size, self.n_hidden).to(device)
        syn_u = self.U * torch.ones(self.batch_size, self.n_hidden).to(device)

        return copy.copy(self.h_init.to(device)), syn_x, syn_u

    # @torch.jit.script_method
    def rnn_cell(
            self,
            stim: torch.Tensor,
            context: torch.Tensor,
            h: torch.Tensor,
            syn_x: torch.Tensor,
            syn_u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update neural activity and short-term synaptic plasticity values"""
        syn_x = syn_x + (self.alpha_x * (1 - syn_x) - self.dt_sec * syn_u * syn_x * h)
        syn_u = syn_u + (self.alpha_u * (self.U - syn_u) + self.dt_sec * self.U * (1 - syn_u) * h)
        syn_x = torch.clip(syn_x, 0.0, 1.0)
        syn_u = torch.clip(syn_u, 0.0, 1.0)
        h_post = syn_u * syn_x * h

        # Update the hidden state.
        stim_input = self.input(stim)
        context_input = self.context(context)
        h = h * (1 - self.alpha_neuron) + self.alpha_neuron * (
            stim_input + context_input + h_post @ self.W + self.b_rec
        ) + self.noise_std * torch.randn(h.size()).to(device=h.device)
        h = self.relu(h)

        return h, syn_x, syn_u

    # @torch.jit.script_method
    def forward(
        self,
        stimulus: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        h, syn_x, syn_u = self._init_before_trial()

        inputs = stimulus.unbind(1)
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



class LSTM(torch.jit.ScriptModule):
    def __init__(self, **kwargs):

        super(LSTM, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._initialize_weights()

    def reset_context_weights(self):
        nn.init.xavier_uniform_(self.context, gain=1.0)

    def _initialize_weights(self):

        super(LSTM, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(4 * self.n_hidden, self.n_stimulus))
        self.context = nn.Parameter(torch.randn(4 * self.n_hidden, self.n_context))
        self.weight_hh = nn.Parameter(torch.randn(4 * self.n_hidden, self.n_hidden))
        self.bias_ih = nn.Parameter(torch.randn(4 * self.n_hidden))
        self.bias_hh = nn.Parameter(torch.randn(4 * self.n_hidden))
        self.output = nn.Linear(self.n_hidden, self.n_output)
        self.value = nn.Linear(self.n_hidden, 1)

        nn.init.xavier_uniform_(self.weight_ih, gain=1.0)
        nn.init.xavier_uniform_(self.context, gain=1.0)
        nn.init.xavier_uniform_(self.weight_hh, gain=1.0)

        self.h0 = nn.Parameter(torch.zeros(1, self.n_hidden))
        self.c0 = nn.Parameter(torch.zeros(1, self.n_hidden))

        self.classifiers = Classifers(n_input=2*self.n_hidden, n_classifiers=3)

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

            gates = (torch.mm(inputs[i], self.weight_ih.t()) + self.bias_ih +
                     torch.mm(hx, self.weight_hh.t()) + self.bias_hh +
                     torch.mm(ctx[i], self.context.t()))

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

