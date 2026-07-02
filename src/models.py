import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PureMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_in_conv = x_in.transpose(1, 2)
        x_in_conv = self.conv1d(x_in_conv)[:, :, :L]
        x_in_silu = F.silu(x_in_conv.transpose(1, 2))

        x_dbl = self.x_proj(x_in_silu)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())

        y_seq = []
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)

        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            A_t = torch.exp(dt_t * A)
            B_t = B_ssm[:, t, :].unsqueeze(1)
            x_t = x_in_silu[:, t, :].unsqueeze(-1)
            dBx_t = (dt_t * x_t) * B_t
            h = A_t * h + dBx_t
            C_t = C_ssm[:, t, :].unsqueeze(1)
            y_t = torch.sum(h * C_t, dim=-1)
            y_seq.append(y_t)

        y = torch.stack(y_seq, dim=1)
        y = y + x_in_silu * self.D

        out = y * F.silu(z)
        return self.out_proj(out)


class PureMambaMotionPredictor(nn.Module):
    def __init__(self, input_dim=2, d_model=128, output_dim=2, future_steps=30, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.future_steps = future_steps
        self.d_model = d_model
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, d_model)

        self.mamba_layers = nn.ModuleList([
            PureMambaBlock(d_model=d_model) for _ in range(num_layers)
        ])

        self.out_proj = nn.Linear(d_model, future_steps * output_dim)

    def forward(self, past_traj):
        B, L, D_in = past_traj.shape
        x = self.input_proj(past_traj)

        for layer in self.mamba_layers:
            x = layer(x)

        last_state = x[:, -1, :]
        future_pred = self.out_proj(last_state)
        future_pred = future_pred.view(B, self.future_steps, self.output_dim)

        return future_pred


class BezierSplineDecoder(nn.Module):
    def __init__(self, num_control_points, num_steps):
        super().__init__()
        self.num_control_points = num_control_points
        self.num_steps = num_steps

        t = torch.linspace(0, 1, num_steps)
        n = num_control_points - 1

        basis = torch.zeros(num_steps, num_control_points)
        for i in range(num_control_points):
            binom = math.comb(n, i)
            basis[:, i] = binom * ((1 - t) ** (n - i)) * (t ** i)

        self.register_buffer('basis', basis)

    def forward(self, control_points):
        return torch.einsum('sc, bcd -> bsd', self.basis, control_points)


class TransformerMotionPredictor(nn.Module):
    def __init__(self, input_dim=2, d_model=128, nhead=4, num_layers=3, future_steps=30, past_steps=20, num_control_points=6, dropout=0.1):
        super().__init__()
        self.future_steps = future_steps
        self.num_control_points = num_control_points
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, past_steps, d_model))

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation='gelu',
            dropout=dropout,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.out_proj = nn.Linear(d_model, num_control_points * 2)
        self.spline_decoder = BezierSplineDecoder(num_control_points, future_steps)

    def forward(self, past_traj):
        x = self.input_proj(past_traj)
        x = x + self.pos_encoder
        x = self.transformer(x)
        last_state = self.final_norm(x[:, -1, :])
        control_points = self.out_proj(last_state).view(-1, self.num_control_points, 2)
        future_pred = self.spline_decoder(control_points)
        return future_pred
