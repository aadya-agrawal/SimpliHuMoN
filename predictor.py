import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) 
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

class block(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        num_heads = 4
        assert in_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.head_dim = in_dim // num_heads
        self.norm_past1 = nn.LayerNorm(in_dim, elementwise_affine= False, eps=1e-6)
        self.norm_past2 = nn.LayerNorm(in_dim, elementwise_affine= False, eps=1e-6)
        self.norm_future1 = nn.LayerNorm(in_dim, elementwise_affine= False, eps=1e-6)
        self.norm_future2 = nn.LayerNorm(in_dim, elementwise_affine= False, eps=1e-6)
        self.qkv_past = nn.Linear(in_dim, in_dim*3)
        self.norm_q_past = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.norm_k_past = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.qkv_future = nn.Linear(in_dim, in_dim*3)
        self.norm_q_future = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.norm_k_future = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out_past = nn.Linear(in_dim,in_dim)
        self.out_future = nn.Linear(in_dim,in_dim)
        hidden_dim = in_dim*2 
        self.mlp_past = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,in_dim)
        )
        self.mlp_future = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,in_dim)
        )

    def forward(self, past, future):    
        B, N_past, C = past.shape
        N_future = future.shape[1]

        past = self.norm_past1(past)
        future = self.norm_future1(future)
        
        past_scaled = past
        future_scaled = future

        qkv_past = self.qkv_past(past_scaled).reshape(B, N_past, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_past, k_past, v_past = qkv_past.unbind(0)
        q_past, k_past = self.norm_q_past(q_past), self.norm_k_past(k_past)
        qkv_future = self.qkv_future(future_scaled).reshape(B, N_future, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_future, k_future, v_future = qkv_future.unbind(0)
        q_future, k_future = self.norm_q_future(q_future), self.norm_k_future(k_future)

        q = torch.cat([q_past, q_future], dim=-2)
        k = torch.cat([k_past, k_future], dim=-2)
        v = torch.cat([v_past, v_future], dim=-2)

        x = nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N_past + N_future, C)

        past_int = self.out_past(x[:, :N_past, :]) 
        future_int = self.out_future(x[:, N_past:, :]) 

        past_int = past_int + past  
        future_int = future_int + future 

        past_int_norm = self.norm_past2(past_int)
        future_int_norm = self.norm_future2(future_int)

        past_int = self.mlp_past(past_int_norm) 
        future_int = self.mlp_future(future_int_norm) 
        
        past = past + past_int 
        future = future + future_int
        
        return past, future

class PoseTrajPredictor(nn.Module):
    def __init__(self, nblocks=16, embed_dim=48, num_joints=15, input_time=25, num_future_steps=50, num_modes=6):
        super(PoseTrajPredictor, self).__init__()
        self.output_time = num_future_steps
        self.num_joints = num_joints
        self.num_modes = num_modes

        self.input_proj = nn.Linear(3, embed_dim, bias=False)
        self.mlp_enc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.inp_enc = PositionalEncoding(embed_dim, max_len=input_time)
        self.traj_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pose_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.future_tokens = nn.Parameter(torch.randn(num_future_steps*2, 3)*1e-3) 
        self.fut_proj = nn.Linear(3, embed_dim, bias=False)
        self.out_enc = PositionalEncoding(embed_dim, max_len=num_future_steps)

        self.layers = nn.ModuleList([block(embed_dim) for _ in range(nblocks)])

        self.multihead_proj = nn.Linear(embed_dim, num_modes*embed_dim)
        self.output_proj = nn.Linear(embed_dim, 3)
        self.mlp_dec = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )

    def forward(self, data):
        BN, T, D = data['input_seq'].shape    
        J = self.num_joints
        past_inp = data['input_seq'].view(BN,T,J,-1) 

        traj_rel = past_inp[:,:,0] - past_inp[:,-1:,0] 
        traj_past = self.inp_enc(self.input_proj(traj_rel))
        traj_past = traj_past + self.traj_type_emb 

        pose_rel = (past_inp - past_inp[:,:,0:1]).reshape(BN,T,D)
        pose_past = self.inp_enc(self.mlp_enc(pose_rel))
        pose_past = pose_past + self.pose_type_emb

        past = torch.cat([traj_past, pose_past], dim=1)

        future_query = self.future_tokens.unsqueeze(0).expand(BN,-1,-1)
        future_query = self.fut_proj(future_query)
        future_traj, future_pose = future_query.chunk(2, dim=1)

        future_traj = self.out_enc(future_traj)
        future_traj = future_traj + self.traj_type_emb

        future_pose = self.out_enc(future_pose)
        future_pose = future_pose + self.pose_type_emb
        
        future = torch.cat([future_traj, future_pose], dim=-2) 

        for layer in self.layers:
            past, future = layer(past, future)

        future = self.multihead_proj(future).view(BN, 2*self.output_time, self.num_modes, -1).permute(2, 0, 1, 3)
        future_traj, future_pose = future.chunk(2, dim=-2)

        pred_traj = self.output_proj(future_traj).reshape(self.num_modes, BN, self.output_time, 1, -1)
        pred_traj = pred_traj + past_inp[:,-1:,0:1].unsqueeze(0)

        pred_pose = self.mlp_dec(future_pose).reshape(self.num_modes, BN, self.output_time, J, -1)

        pred = pred_pose + pred_traj
        return pred
    

class TrajectoryPredictor(nn.Module):
    def __init__(self, nblocks=16, embed_dim=48, num_joints=1, input_time=8, num_future_steps=12, num_modes=20):
        super(TrajectoryPredictor, self).__init__()
        self.output_time = num_future_steps
        self.num_modes = num_modes

        self.input_proj = nn.Linear(2, embed_dim, bias=False)
        self.inp_enc = PositionalEncoding(embed_dim, max_len=input_time)

        self.future_tokens = nn.Parameter(torch.randn(num_future_steps, 2)*1e-3) 
        self.fut_proj = nn.Linear(2, embed_dim, bias=False)
        self.out_enc = PositionalEncoding(embed_dim, max_len=num_future_steps)

        self.layers = nn.ModuleList([block(embed_dim) for _ in range(nblocks)])

        self.multihead_proj = nn.Linear(embed_dim, num_modes*embed_dim)
        self.output_proj = nn.Linear(embed_dim, 2)

    def forward(self, data):
        BN = data['input_seq'].shape[0]
        past_inp = data['input_seq'] #.view(BN,T,J,-1) 

        traj_rel = past_inp - past_inp[:,-1:] 
        traj_past = self.inp_enc(self.input_proj(traj_rel))
        past = traj_past

        future_query = self.future_tokens.unsqueeze(0).expand(BN,-1,-1)
        future_query = self.fut_proj(future_query)
        future = self.out_enc(future_query)

        for layer in self.layers:
            past, future = layer(past, future)

        future = self.multihead_proj(future).view(BN, self.output_time, self.num_modes, -1).permute(2, 0, 1, 3)

        pred_traj = self.output_proj(future).reshape(self.num_modes, BN, self.output_time, -1)
        pred = pred_traj + past_inp[:,-1:].unsqueeze(0)
        return pred
    
class PosePredictor(nn.Module):
    def __init__(self, nblocks=16, embed_dim=48, num_joints=16, input_time=25, num_future_steps=50, num_modes=7):
        super(PosePredictor, self).__init__()
        self.output_time = num_future_steps
        self.num_joints = num_joints
        self.num_modes = num_modes

        self.mlp_enc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.inp_enc = PositionalEncoding(embed_dim, max_len=input_time)

        self.future_tokens = nn.Parameter(torch.randn(num_future_steps, 3)*1e-3) 
        self.fut_proj = nn.Linear(3, embed_dim, bias=False)
        self.out_enc = PositionalEncoding(embed_dim, max_len=num_future_steps)

        self.layers = nn.ModuleList([block(embed_dim) for _ in range(nblocks)])

        self.multihead_proj = nn.Linear(embed_dim, num_modes*embed_dim)
        self.mlp_dec = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )

    def forward(self, data):
        BN, T, D = data['input_seq'].shape    
        J = self.num_joints
        past_inp = data['input_seq'].view(BN,T,J,-1) 

        pose_rel = (past_inp - past_inp[:,:,0:1]).reshape(BN,T,D)
        pose_past = self.inp_enc(self.mlp_enc(pose_rel))
        past = pose_past

        future_query = self.future_tokens.unsqueeze(0).expand(BN,-1,-1)
        future_query = self.fut_proj(future_query)
        future_pose = self.out_enc(future_query)        
        future = future_pose

        for layer in self.layers:
            past, future = layer(past, future)

        future = self.multihead_proj(future).view(BN, self.output_time, self.num_modes, -1).permute(2, 0, 1, 3)
        pred = self.mlp_dec(future).reshape(self.num_modes, BN, self.output_time, J, -1)
        return pred