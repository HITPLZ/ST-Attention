import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe[:170, :]  # 截取与节点数相匹配的部分
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return x + self.pe

# Z gate
class FusionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FusionModel, self).__init__()
        self.linear_layer = nn.Linear(input_dim * 2, output_dim)

    def forward(self, temporal_input, spatial_input):
        z = torch.sigmoid(self.linear_layer(torch.cat((temporal_input, spatial_input), dim=-1)))
        fused_output = z * temporal_input + (1 - z) * spatial_input
        return fused_output


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value,):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)

        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        #torch.save(out, 'attn_score_before.pt')
        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y=None, dim=-2, c=None, bat=None):
        x = x.transpose(dim, -2)

        # x: (batch_size, ..., length, model_dim)
        if c is not None:
            residual = c
        else:
            residual = x
        if y is None:
            out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
            # if bat == 1115:
            #     torch.save(out, 'attn_score_after.pt')
        else:
            y = y.transpose(dim, -2)
            out = self.attn(y, x, x)
            # if bat == 1115:
            #     torch.save(out, 'crs_attn_score_after.pt')
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class adjmx_crsAttn_funsion_04(nn.Module):
    """
    Paper: STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
    Link: https://arxiv.org/abs/2308.10425
    Official Code: https://github.com/XDZhelheim/STAEformer
    """

    def __init__(
            self,
            num_nodes,
            # adj_mx,
            in_steps,
            out_steps,
            steps_per_day,
            input_dim,
            output_dim,
            input_embedding_dim,
            tod_embedding_dim,
            ts_embedding_dim,
            dow_embedding_dim,
            adaptive_embedding_dim,
            graph_proj_dim,
            graph_proj_hidden_dim,
            feed_forward_dim,
            num_heads,
            num_layers,
            num_layers_m,
            dropout,
            use_mixed_proj,
            bat
    ):
        super().__init__()
        # self.adj_mx = adj_mx
        self.bat = bat
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.ts_embedding_dim = ts_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.graph_proj_dim = graph_proj_dim
        self.graph_proj_hidden_dim = graph_proj_hidden_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + adaptive_embedding_dim
                + ts_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.num_layers_m = num_layers_m
        if self.input_embedding_dim > 0:
            self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        ####################

        self.attn_layers_m2 = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers_m)
            ]
        )

        self.attn_layers_c = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            ]
        )

        if self.ts_embedding_dim > 0:
            self.time_series_emb_layer = nn.Conv2d(
                in_channels=self.input_dim * self.in_steps, out_channels=self.ts_embedding_dim, kernel_size=(1, 1),
                bias=True)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        # self.bat += 1

        x = history_data
        batch_size, _, num_nodes, _ = x.shape
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        if self.ts_embedding_dim > 0:
            input_data = x.transpose(1, 2).contiguous()
            input_data = input_data.view(
                batch_size, self.num_nodes, -1).transpose(1, 2).unsqueeze(-1)
            # B L*3 N 1
            time_series_emb = self.time_series_emb_layer(input_data)
            time_series_emb = time_series_emb.transpose(1, -1).expand(batch_size, self.in_steps, self.num_nodes,
                                                                      self.ts_embedding_dim)
        # B ts_embedding_dim N 1

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]

        if self.ts_embedding_dim > 0:
            features.append(time_series_emb)
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        temporal_x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        spatial_x = temporal_x.clone()

        for index, (attn_t, attn_s) in enumerate(zip(self.attn_layers_t, self.attn_layers_s)):
            temporal_x = attn_t(temporal_x, dim=1, bat=self.bat)
            spatial_x = attn_s(spatial_x, dim=2)

        for attn in self.attn_layers_c:
            x = attn(temporal_x, spatial_x, dim=2, bat=self.bat)

        for attn in self.attn_layers_m2:
            x = attn(x, dim=2)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)
        # if self.bat == 1115: self.bat = 0
        return out
