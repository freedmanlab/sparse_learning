import torch
from torch import nn
from typing import List, Literal, Tuple

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


