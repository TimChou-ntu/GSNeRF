import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1.0 / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(stdv, stdv)


def masked_softmax(x, mask, **kwargs):
    x_masked = x.masked_fill(mask == 0, -float("inf"))

    return torch.softmax(x_masked, **kwargs)


## Auto-encoder network
class ConvAutoEncoder(nn.Module):
    def __init__(self, num_ch, S):
        super(ConvAutoEncoder, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_ch, num_ch * 2, 3, stride=1, padding=1),
            # nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch * 4, 3, stride=1, padding=1),
            # nn.LayerNorm(S // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_ch * 4, num_ch * 4, 3, stride=1, padding=1),
            # nn.LayerNorm(S // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )

        # Decoder
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch * 4, 4, stride=2, padding=1),
            # nn.LayerNorm(S // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 8, num_ch * 2, 4, stride=2, padding=1),
            # nn.LayerNorm(S // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch, 4, stride=2, padding=1),
            # nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        # Output
        self.conv_out = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch, 3, stride=1, padding=1),
            # nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        conv1_out = x
        x = self.conv2(x)
        conv2_out = x
        x = self.conv3(x)

        x = self.t_conv1(x)
        x = self.t_conv2(torch.cat([x, conv2_out], dim=1))
        x = self.t_conv3(torch.cat([x, conv1_out], dim=1))

        x = self.conv_out(torch.cat([x, input], dim=1))

        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = masked_softmax(attn, mask, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x += residual

        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.transpose(1, 2).unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Renderer(nn.Module):
    def __init__(self, nb_samples_per_ray, nb_view=8, nb_class=21, 
                 using_semantic_global_tokens=True, only_using_semantic_global_tokens=True, use_batch_semantic_feature=False):
        super(Renderer, self).__init__()

        self.nb_view = nb_view
        self.nb_class = nb_class
        self.using_semantic_global_tokens = using_semantic_global_tokens
        self.only_using_semantic_global_tokens = only_using_semantic_global_tokens
        self.dim = 32

        if use_batch_semantic_feature:
            self.nb_class = self.nb_class * 9
        else:
            self.nb_class = self.nb_class
            
        self.semantic_token_gen = nn.Linear(1 + self.nb_class, self.dim)

        self.attn_token_gen = nn.Linear(24 + 1 + 8, self.dim)

        ## Self-Attention Settings
        d_inner = self.dim
        n_head = 4
        d_k = self.dim // n_head
        d_v = self.dim // n_head
        num_layers = 4
        self.attn_layers = nn.ModuleList(
            [
                EncoderLayer(self.dim, d_inner, n_head, d_k, d_v)
                for i in range(num_layers)
            ]
        )

        self.semantic_attn_layers = nn.ModuleList(
            [
                EncoderLayer(self.dim, d_inner, n_head, d_k, d_v)
                for i in range(num_layers)
            ]
        )

        # +1 because we add the mean and variance of input features as global features
        if using_semantic_global_tokens and only_using_semantic_global_tokens:
            self.semantic_dim = self.dim
            ## Processing the mean and variance of semantic features
            self.semantic_var_mean_fc1 = nn.Linear(self.nb_class*2, self.dim)
            self.semantic_var_mean_fc2 = nn.Linear(self.dim, self.dim)
        elif using_semantic_global_tokens:
            self.semantic_dim = self.dim * (nb_view + 1)
            ## Processing the mean and variance of semantic features
            self.semantic_var_mean_fc1 = nn.Linear(self.nb_class*2, self.dim)
            self.semantic_var_mean_fc2 = nn.Linear(self.dim, self.dim)
        else:
            self.semantic_dim = self.dim * nb_view

        self.semantic_fc1 = nn.Linear(self.semantic_dim, self.semantic_dim)
        self.semantic_fc2 = nn.Linear(self.semantic_dim, self.semantic_dim // 2)
        self.semantic_fc3 = nn.Linear(self.semantic_dim // 2, nb_class)

        ## Processing the mean and variance of input features
        self.var_mean_fc1 = nn.Linear(16, self.dim)
        self.var_mean_fc2 = nn.Linear(self.dim, self.dim)


        ## Setting mask of var_mean always enabled
        self.var_mean_mask = torch.tensor([1])
        self.var_mean_mask.requires_grad = False

        ## For aggregating data along ray samples
        self.auto_enc = ConvAutoEncoder(self.dim, nb_samples_per_ray)

        self.sigma_fc1 = nn.Linear(self.dim, self.dim)
        self.sigma_fc2 = nn.Linear(self.dim, self.dim // 2)
        self.sigma_fc3 = nn.Linear(self.dim // 2, 1)


        self.rgb_fc1 = nn.Linear(self.dim + 9, self.dim)
        self.rgb_fc2 = nn.Linear(self.dim, self.dim // 2)
        self.rgb_fc3 = nn.Linear(self.dim // 2, 1)

        ## Initialization
        self.sigma_fc1.apply(weights_init)
        self.sigma_fc2.apply(weights_init)
        self.sigma_fc3.apply(weights_init)
        self.rgb_fc1.apply(weights_init)
        self.rgb_fc2.apply(weights_init)
        self.rgb_fc3.apply(weights_init)

    def forward(self, viewdirs, feat, occ_masks, middle_pts_mask):
        ## Viewing samples regardless of batch or ray
        N, S, V = feat.shape[:3]
        feat = feat.view(-1, *feat.shape[2:])
        v_feat = feat[..., :24]
        s_feat = feat[..., 24 : 24 + 8]
        colors = feat[..., 24 + 8 : 24 + 8 + 3]
        semantic_feat = feat[..., 24 + 8 + 3 : -1]
        vis_mask = feat[..., -1:].detach()

        occ_masks = occ_masks.view(-1, *occ_masks.shape[2:])
        viewdirs = viewdirs.view(-1, *viewdirs.shape[2:])

        ## Mean and variance of 2D features provide view-independent tokens
        var_mean = torch.var_mean(s_feat, dim=1, unbiased=False, keepdim=True)
        var_mean = torch.cat(var_mean, dim=-1)
        var_mean = F.elu(self.var_mean_fc1(var_mean))
        var_mean = F.elu(self.var_mean_fc2(var_mean))

        ## Converting the input features to tokens (view-dependent) before self-attention
        tokens = F.elu(
            self.attn_token_gen(torch.cat([v_feat, vis_mask, s_feat], dim=-1))
        )
        tokens = torch.cat([tokens, var_mean], dim=1)

        # by adding middle_pts_mask, we can only take the predicted depth's points into account

        if self.using_semantic_global_tokens:
            semantic_var_mean = torch.var_mean(semantic_feat[middle_pts_mask.view(-1)], dim=1, unbiased=False, keepdim=True)
            semantic_var_mean = torch.cat(semantic_var_mean, dim=-1)
            semantic_var_mean = F.elu(self.semantic_var_mean_fc1(semantic_var_mean))
            semantic_var_mean = F.elu(self.semantic_var_mean_fc2(semantic_var_mean))
        # (N_rays, 1, views, feat_dim)
        semantic_tokens = F.elu(
            self.semantic_token_gen(torch.cat([semantic_feat[middle_pts_mask.view(-1)], vis_mask[middle_pts_mask.view(-1)]], dim=-1))
        )

        if self.using_semantic_global_tokens:
            semantic_tokens = torch.cat([semantic_tokens, semantic_var_mean], dim=1)

        ## Adding a new channel to mask for var_mean
        vis_mask = torch.cat(
            [vis_mask, self.var_mean_mask.view(1, 1, 1).expand(N * S, -1, -1).to(vis_mask.device)], dim=1
        )
        ## If a point is not visible by any source view, force its masks to enabled
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 1, 1)

        ## Taking occ_masks into account, but remembering if there were any visibility before that
        mask_cloned = vis_mask.clone()
        vis_mask[:, :-1] *= occ_masks
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 1, 1)
        masks = vis_mask * mask_cloned

        ## Performing self-attention
        for layer in self.attn_layers:
            tokens, _ = layer(tokens, masks)

        for semantic_layer in self.semantic_attn_layers:
            if self.using_semantic_global_tokens:
                # mask has shape (N_rays*N_points, nb_views+1, 1), because of the var_mean_mask, semantic not using that
                semantic_tokens, _ = semantic_layer(semantic_tokens, masks[middle_pts_mask.view(-1)])
            else:
                semantic_tokens, _ = semantic_layer(semantic_tokens, masks[middle_pts_mask.view(-1)][:, :-1])

        ## Predicting sigma with an Auto-Encoder and MLP
        sigma_tokens = tokens[:, -1:]
        sigma_tokens = sigma_tokens.view(N, S, self.dim).transpose(1, 2)
        sigma_tokens = self.auto_enc(sigma_tokens)
        sigma_tokens = sigma_tokens.transpose(1, 2).reshape(N * S, 1, self.dim)

        sigma_tokens_ = F.elu(self.sigma_fc1(sigma_tokens))
        sigma_tokens_ = F.elu(self.sigma_fc2(sigma_tokens_))
        # sigma shape (N_rays*N_points, 1)
        sigma = torch.relu(self.sigma_fc3(sigma_tokens_[:, 0]))

        if self.using_semantic_global_tokens and self.only_using_semantic_global_tokens:
            semantic_global_tokens = semantic_tokens[:, -1:]
        elif self.using_semantic_global_tokens:
            semantic_global_tokens = semantic_tokens.reshape(-1, self.semantic_dim)
        else:
            semantic_global_tokens = semantic_tokens.reshape(-1, self.semantic_dim)
        semantic_tokens_ = F.elu(self.semantic_fc1(semantic_global_tokens))
        semantic_tokens_ = F.elu(self.semantic_fc2(semantic_tokens_))
        semantic_tokens_ = torch.relu(self.semantic_fc3(semantic_tokens_))
  
        semantic = semantic_tokens_.reshape(N, -1).unsqueeze(1)

        ## Concatenating positional encodings and predicting RGB weights
        rgb_tokens = torch.cat([tokens[:, :-1], viewdirs], dim=-1)
        rgb_tokens = F.elu(self.rgb_fc1(rgb_tokens))
        rgb_tokens = F.elu(self.rgb_fc2(rgb_tokens))
        rgb_w = self.rgb_fc3(rgb_tokens)
        rgb_w = masked_softmax(rgb_w, masks[:, :-1], dim=1)

        rgb = (colors * rgb_w).sum(1)

        outputs = torch.cat([rgb, sigma], -1)
        outputs = outputs.reshape(N, S, -1)

        return outputs, semantic


class Semantic_predictor(nn.Module):
    def __init__(self, nb_view=6, nb_class=0):
        super(Semantic_predictor, self).__init__()
        self.nb_class = nb_class
        self.dim = 32
        # self.attn_token_gen = nn.Linear(24 + 1 + self.nb_class, self.dim)
        self.attn_token_gen = nn.Linear(1 + self.nb_class, self.dim)
        self.semantic_dim = self.dim * nb_view

        # Self-Attention Settings, This attention is cross-view attention for a point, which represent a pixel in target view
        d_inner = self.dim
        n_head = 4
        d_k = self.dim // n_head
        d_v = self.dim // n_head
        num_layers = 4
        self.attn_layers = nn.ModuleList(
            [
                EncoderLayer(self.dim, d_inner, n_head, d_k, d_v)
                for i in range(num_layers)
            ]
        )
        self.semantic_fc1 = nn.Linear(self.semantic_dim, self.semantic_dim)
        self.semantic_fc2 = nn.Linear(self.semantic_dim, self.semantic_dim // 2)
        self.semantic_fc3 = nn.Linear(self.semantic_dim // 2, nb_class)

    def forward(self, feat, occ_masks):
        if feat.dim() == 3:
            feat = feat.unsqueeze(1)
        if occ_masks.dim() == 3:
            occ_masks = occ_masks.unsqueeze(1)
        N, S, V, C = feat.shape # (num_rays, num_samples, num_views, feat_dim), S should be 1 here

        feat = feat.view(-1, *feat.shape[2:]) # (num_rays * num_samples, num_views, feat_dim)
        v_feat = feat[..., :24]
        s_feat = feat[..., 24 : 24 + self.nb_class]
        colors = feat[..., 24 + self.nb_class : -1]
        vis_mask = feat[..., -1:].detach()

        occ_masks = occ_masks.view(-1, *occ_masks.shape[2:])

        tokens = F.elu(
            # self.attn_token_gen(torch.cat([v_feat, vis_mask, s_feat], dim=-1))
            self.attn_token_gen(torch.cat([vis_mask, s_feat], dim=-1))
        )

        ## If a point is not visible by any source view, force its masks to enabled
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 0, 1)

        ## Taking occ_masks into account, but remembering if there were any visibility before that
        mask_cloned = vis_mask.clone()
        vis_mask *= occ_masks
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 0, 1)
        masks = vis_mask * mask_cloned

        ## Performing self-attention on source view features, 
        for layer in self.attn_layers:
            tokens, _ = layer(tokens, masks)
            # tokens, _ = layer(tokens, vis_mask)

        ## Predicting semantic with MLP
        ## tokens shape: (N*S, V, dim), S = 1
        tokens = tokens.reshape(N, V*self.dim)
        semantic_tokens_ = F.elu(self.semantic_fc1(tokens))
        semantic_tokens_ = F.elu(self.semantic_fc2(semantic_tokens_))
        semantic_tokens_ = torch.relu(self.semantic_fc3(semantic_tokens_))

        semantic = semantic_tokens_.reshape(N, S, -1)

        return semantic