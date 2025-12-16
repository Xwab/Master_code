import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from .quant_utils import Quantizer
from .hadamard_utils import apply_hadamard
def get_image(tensor, name):
    # 对张量进行处理
    tensor = tensor.view(-1)
    t = tensor.cpu().numpy()
    # 创建柱形图
    plt.bar(range(len(t)), t)

    # 设置图表标题和坐标轴标签
    plt.title('Bar Chart of Processed Tensor')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 获取当前时间并格式化为字符串
    # 图片存储路径和文件名
    image_dir = 'image_tensor'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_path = os.path.join(image_dir, f'{name}.png')
    print(image_path)

    # 保存图表为图片
    plt.savefig(image_path)

    # 关闭当前图形窗口
    plt.close()

def cal_out(matrix):
    return torch.sqrt(torch.mean(torch.pow(matrix, 2),dim=-1))

def matrix_solve(D, B, T):
    # p = seqlen, n = hiddensize, m = 1024
    p, n = D.shape
    m, _ = B.shape

    U_D, s_D, Vh_D = torch.linalg.svd(D, full_matrices=False)
    U_B, s_B, Vh_B = torch.linalg.svd(B, full_matrices=False)

    T_transformed = U_D.t() @ T @ U_B  # 形状 (p, m)
    s_D_col = s_D.unsqueeze(1)  # 形状 (p, 1)
    s_B_row = s_B.unsqueeze(0)  # 形状 (1, m)
    denominator = s_D_col @ s_B_row  # 外积，形状 (p, m)

    denominator = torch.where(denominator != 0, denominator, torch.tensor(1e-10))
    Y = T_transformed / denominator

    S_T = Vh_D.T @ Y @ Vh_B
    
    return S_T.T


class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV", quant = False ,nbits = 16, group_size = 0, sym=True, clip_ratio=1.0) -> None:
        super().__init__()
        self.quant = quant
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)
        self.gate = nn.Parameter(torch.ones(1, U.size(1)).to(self.ALinear.weight.data.dtype))


        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)
        self.truncation_rank = S.size(0)
        if sigma_fuse == "UV":
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == "U":
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == "V":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()
        elif sigma_fuse == "None":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        #self.quantizer = Quantizer(nbits, group_size, sym, clip_ratio)

    @staticmethod
    def from_linear_matrix(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        alphaq=1,
        sigma_fuse="UV",
        rank_align=1,
        middle=False,
        name=None,
        whiten=False,
        quant=False
    ):
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        assert ic_split == 1 or oc_split == 1
        # compressed_params = int(n_params * param_ratio)
        #rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(rank)
        print(rank)

        w = linear.weight.data.float()
        w = w.to(torch.float32)
        w_ori = w
        if act_aware:
            in_scale_matrix = 1  # avoid zero division
            out_scale_matrix = 1
            scale_exp_in = None
            scale_exp_out = None
            if hasattr(linear, "in_scale_matrix"):
                in_scale_matrix *= linear.in_scale_matrix
                w = torch.matmul(w, in_scale_matrix.to(torch.float32))
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")
                if whiten == False:
                    in_scale_inv = torch.linalg.inv(in_scale_matrix.transpose(-1, -2).to(torch.float32))
                else:
                    in_scale_inv = linear.in_scale_inv_matrix.to(torch.float32).t()
            if hasattr(linear, "out_scale_matrix"):
                print("do_out_scale")
                out_scale_matrix *= linear.out_scale_matrix.view(-1,1)
                w *= out_scale_matrix


        Us = []
        Ss = []
        Vs = []
        if (torch.isnan(w).any()):
            print("nan")
        if (torch.isinf(w).any()):
            print("inf")
        U_, S_, V_ = torch.linalg.svd(w, full_matrices=False)
        U = U_[:,:rank]
        S = S_[:rank]
        V = V_[:rank,:].t()
        if act_aware:
            if hasattr(linear, "in_scale_matrix"):
                V = torch.matmul(in_scale_inv.to(torch.float32), V)
            if hasattr(linear, "out_scale_matrix"):
                U /= out_scale_matrix
        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        new_linear = SVDLinear(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V.to(linear.weight.dtype), bias, sigma_fuse, quant=quant)
        new_linear.to(linear.weight.dtype)
        new_linear.to(device=linear.weight.device)
        new_linear.truncation_rank = rank

        return new_linear
    
    @staticmethod
    def from_linear_matrix_asvd(
        linear: nn.Linear,
        rank: float,
        act_aware=True,
        ic_split=1,
        oc_split=1,
        alpha=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        # print("rank", rank)
        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division
            w = w * scaling_diag_matrix.view(1, -1)
        Us = []
        Ss = []
        Vs = []
        #try:
        U, S, V = torch.svd_lowrank(w, q=rank)
        #except:
        #    print(f"svd failed for {linear}, disable act_aware")
        #   return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
        if act_aware:
            V = V / scaling_diag_matrix.view(-1, 1)
        Us = [U]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        for S in Ss:
            if (S != S).any():
                print("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for U in Us:
            if (U != U).any():
                print("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                print("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )

        assert len(Us) == len(Ss) == len(Vs) == 1
        new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        return new_linear
    
    @staticmethod
    def from_linear_matrix_svd(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()
        rank = int(rank)
        w = linear.weight.data.float()
        Us = []
        Ss = []
        Vs = []
        #try:
        U, S, V = torch.svd_lowrank(w, q=rank)
        #except:
        #    print(f"svd failed for {linear}, disable act_aware")
        #   return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
        
        Us = [U]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        for S in Ss:
            if (S != S).any():
                print("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for U in Us:
            if (U != U).any():
                print("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                print("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )

        assert len(Us) == len(Ss) == len(Vs) == 1
        new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse)
        new_linear.to(linear.weight.dtype).to(linear.weight.device)
        return new_linear
    
    def fused_hadamard_matrix(self):
        VT_weight = self.BLinear.weight.data
        VT_weight = apply_hadamard(VT_weight.t())
        U_weight = self.ALinear.weight.data
        U_weight = apply_hadamard(U_weight)
        self.BLinear.weight.data = VT_weight
        self.ALinear.weight.data = U_weight


    def update_scaling_matrix(
        self,
        in_scale_matrix_list,
        num_kv_heads = 8,
        hidden_dim = 4096,
        head_dim = 128,    
    ):
        rank = self.truncation_rank
        V_t = self.BLinear.weight.data
        U = self.ALinear.weight.data
        ori_linear_weight = torch.matmul(U, V_t)
        w_list = []
        for i in range(len(in_scale_matrix_list)):
            in_scale_matrix_i = in_scale_matrix_list[i].to(ori_linear_weight.device)
            w_i = ori_linear_weight[i*head_dim:(i+1)*head_dim, :]
            w_i = torch.mm(w_i, in_scale_matrix_i.to(torch.float32))
            w_list.append(w_i)
        w = torch.cat(w_list, dim = 0)
        U_tmp, S_tmp, V_tmp = torch.linalg.svd(w, full_matrices=False)
        U_tmp = U_tmp[:,:rank]
        S_tmp = S_tmp[:rank]
        V_tmp = V_tmp[:rank,:]
        w_lowrank = torch.matmul(U_tmp, torch.diag(S_tmp)).mm(V_tmp) # (num_kv_heads * head_dim, hidden_dim)

        w_target_ori_lowrank = []
        for i in range(num_kv_heads):
            in_scale_inv_i = torch.linalg.inv(in_scale_matrix_list[i].to(torch.float32).to(w.device))
            w_lowrank_i = w_lowrank[i*head_dim:(i+1)*head_dim,:]
            w_target_ori_lowrank.append(torch.mm(w_lowrank_i, in_scale_inv_i))
        w_target_ori_lowrank = torch.cat(w_target_ori_lowrank, dim=0)
        
        in_scale_inv = torch.linalg.lstsq(w, w_target_ori_lowrank).solution
        P, S, Q = torch.linalg.svd(w, full_matrices=False) 
        Q = Q.t()
        Q = torch.matmul(in_scale_inv.to(torch.float32).t(), Q)
        self.ALinear.weight.data = P.mul(S.sqrt()).contiguous()
        self.BLinear.weight.data = Q.t().mul(S.sqrt().view(-1, 1)).contiguous()


    @staticmethod
    def from_linear_matrix_v2(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        num_kv_heads=8,
        hidden_dim=4096,
        head_dim = 128,
        sigma_fuse="UV",
        rank_align=1,
        middle=False,
        name=None,
        args=None,
    ):
        
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        assert ic_split == 1 or oc_split == 1
        # compressed_params = int(n_params * param_ratio)
        #rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(rank)
        print(rank)

        w = linear.weight.data.float()
        w = w.to(torch.float32)
        if act_aware:
            in_scale_matrix_list = [1 for _ in range(num_kv_heads)]  # avoid zero division
            out_scale_matrix = 1
            scale_exp_in = None
            scale_exp_out = None
            w_list = []
            w_ori = w

            if hasattr(linear, "in_scale_matrix"):
                #mean_in_scale_maxtrix = torch.mean(torch.stack(linear.in_scale_matrix), dim = 0)
                #w_target = torch.matmul(w_ori, mean_in_scale_maxtrix.to(w_ori.device).to(torch.float32))

                for i in range(num_kv_heads):
                    in_scale_matrix_list[i] = linear.in_scale_matrix[i].to(w_ori.device) # (hidden_dim, hidden_dim)
                    w_i = w[i*head_dim:(i+1)*head_dim,:] #(head_dim, hidden_dim)
                    #print(w_i.shape, linear.in_scale_matrix[i].shape)
                    w_i = torch.matmul(w_i, in_scale_matrix_list[i].to(torch.float32)) #(head_dim, hidden_dim)
                    w_list.append(w_i)
                w = torch.cat(w_list, dim=0) # (num_kv_heads * head_dim, hidden_dim)
                #w_ori = w
                #print('shape', w.shape)
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")




                '''U_tmp, S_tmp, V_tmp = torch.linalg.svd(w, full_matrices=False)
                U_tmp = U_tmp[:,:rank]
                S_tmp = S_tmp[:rank]
                V_tmp = V_tmp[:rank,:]
                w_lowrank = torch.matmul(U_tmp, torch.diag(S_tmp)).mm(V_tmp) # (num_kv_heads * head_dim, hidden_dim)'''

                '''U_tmp_ori, S_tmp_ori, V_tmp_ori = torch.linalg.svd(w_ori, full_matrices=False)
                U_tmp_ori = U_tmp_ori[:,:rank]
                S_tmp_ori = S_tmp_ori[:rank]
                V_tmp_ori = V_tmp_ori[:rank,:]
                w_ori_lowrank = torch.matmul(U_tmp_ori, torch.diag(S_tmp_ori)).mm(V_tmp_ori) # (num_kv_heads * head_dim, hidden_dim)
                '''
                '''U_tmp_ori, S_tmp_ori, V_tmp_ori = torch.linalg.svd(w_target, full_matrices=False)
                U_tmp_ori = U_tmp_ori[:,:rank]
                S_tmp_ori = S_tmp_ori[:rank]
                V_tmp_ori = V_tmp_ori[:rank,:]
                w_target_lowrank = torch.matmul(U_tmp_ori, torch.diag(S_tmp_ori)).mm(V_tmp_ori)'''
                #方案一
                '''w_target_ori_lowrank = []
                for i in range(num_kv_heads):
                    in_scale_inv_i = torch.linalg.inv(in_scale_matrix_list[i].to(torch.float32).to(w.device))
                    w_lowrank_i = w_lowrank[i*head_dim:(i+1)*head_dim,:]
                    w_target_ori_lowrank.append(torch.mm(w_lowrank_i, in_scale_inv_i))
                w_target_ori_lowrank = torch.cat(w_target_ori_lowrank, dim=0) #(num_kv_heads * head_dim, hidden_dim)
                    
                #方案二
                V_target_ori_lowrank = []
                for i in range(num_kv_heads):
                    in_scale_inv_i = torch.linalg.inv(in_scale_matrix_list[i].to(torch.float32).to(w.device))
                    V_target_ori_lowrank.append(torch.mm(V_tmp, in_scale_inv_i))
                V_target_ori_lowrank = torch.cat(V_target_ori_lowrank, dim=0) #(num_kv_heads * rank, hidden_dim)'''
                #V_ori_lowrank = torch.cat([V_tmp for _ in range(num_kv_heads)], dim=0) #(num_kv_heads * rank, hidden_dim)

                    
                #in_scale_inv_ori = torch.linalg.inv(mean_in_scale_maxtrix.to(torch.float32).to(w.device))
                #w_target_ori_lowrank = torch.matmul(w_target_lowrank, in_scale_inv_ori)

                if args.v_aware:
                    activation_value = torch.mm(linear.in_hidden_matrix, w_ori.T)
                    in_scale_inv = matrix_solve(linear.in_hidden_matrix, w, activation_value)
                else:
                    in_scale_inv = torch.linalg.lstsq(w, w_ori).solution
                    #方案一
                    #in_scale_inv = torch.linalg.lstsq(w, w_target_ori_lowrank).solution
                    #方案二
                    #in_scale_inv = torch.linalg.lstsq(V_ori_lowrank, V_target_ori_lowrank).solution


        if (torch.isnan(w).any()):
            print("nan")
        if (torch.isinf(w).any()):
            print("inf")
        U_, S_, V_ = torch.linalg.svd(w, full_matrices=False)
        U = U_[:,:rank]
        S = S_[:rank]
        V = V_[:rank,:].t() # S_inv * V * sigma * U^T
        if act_aware:
            if hasattr(linear, "in_scale_matrix"):
                V = torch.matmul(in_scale_inv.to(torch.float32).t(), V)

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        new_linear = SVDLinear(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V.to(linear.weight.dtype), bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        new_linear.to(device=linear.weight.device)
        new_linear.truncation_rank = rank

        return new_linear

    @staticmethod
    def from_linear_matrix_v3(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        alphaq=1,
        sigma_fuse="UV",
        rank_align=1,
        middle=False,
        name=None,
        whiten=False,
        args = None,
    ):
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        assert ic_split == 1 or oc_split == 1
        # compressed_params = int(n_params * param_ratio)
        #rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(rank)
        print(rank)

        w = linear.weight.data.float()
        w = w.to(torch.float32)
        if act_aware:
            in_scale_matrix = 1  # avoid zero division
            out_scale_matrix = 1
            scale_exp_in = None
            scale_exp_out = None
            if hasattr(linear, "out_scale_matrix"):
                out_scale_matrix *= linear.out_scale_matrix
                w = torch.matmul(out_scale_matrix.to(torch.float32), w)
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")
                if args.decomposition == 'svd':
                    out_scale_inv = out_scale_matrix.to(torch.float32).t()
                else:
                    out_scale_inv = torch.linalg.inv(out_scale_matrix.to(torch.float32))

            if hasattr(linear, "in_scale_matrix"):
                in_scale_matrix *= linear.in_scale_matrix
                w = torch.matmul(w, in_scale_matrix.to(torch.float32))
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")

                if args.decomposition == 'svd':
                    in_scale_inv = in_scale_matrix.to(torch.float32).t()
                else:
                    in_scale_inv = torch.linalg.inv(in_scale_matrix.transpose(-1, -2).to(torch.float32))
                
                
            #if hasattr(linear, "out_scale_matrix"):
            #    print("do_out_scale")
            #    out_scale_matrix *= linear.out_scale_matrix.view(-1,1)
            #    w *= out_scale_matrix


        Us = []
        Ss = []
        Vs = []
        if (torch.isnan(w).any()):
            print("nan")
        if (torch.isinf(w).any()):
            print("inf")
        U_, S_, V_ = torch.linalg.svd(w, full_matrices=False)
        U = U_[:,:rank]
        S = S_[:rank]
        V = V_[:rank,:].t()
        if act_aware:
            if hasattr(linear, "in_scale_matrix"):
                V = torch.matmul(in_scale_inv.to(torch.float32), V)
            if hasattr(linear, "out_scale_matrix"):
                U = torch.mm(out_scale_inv, U)
        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        new_linear = SVDLinear(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V.to(linear.weight.dtype), bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        new_linear.to(device=linear.weight.device)
        new_linear.truncation_rank = rank

        return new_linear
    

    def from_linear_matrix_v4(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        num_kv_heads=8,
        hidden_dim=4096,
        head_dim = 128,
        sigma_fuse="UV",
        rank_align=1,
        middle=False,
        name=None,
        args=None,
    ):
        
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        assert ic_split == 1 or oc_split == 1
        # compressed_params = int(n_params * param_ratio)
        #rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(rank // 8)
        print(rank)

        w = linear.weight.data.float()
        w = w.to(torch.float32)
        if act_aware:
            in_scale_matrix_list = [1 for _ in range(num_kv_heads)]  # avoid zero division
            out_scale_matrix = 1
            scale_exp_in = None
            scale_exp_out = None
            w_list = []
            w_ori = w

            if hasattr(linear, "in_scale_matrix"):
                #mean_in_scale_maxtrix = torch.mean(torch.stack(linear.in_scale_matrix), dim = 0)
                #w_target = torch.matmul(w_ori, mean_in_scale_maxtrix.to(w_ori.device).to(torch.float32))

                for i in range(num_kv_heads):
                    in_scale_matrix_list[i] = linear.in_scale_matrix[i].to(w_ori.device) # (hidden_dim, hidden_dim)
                    w_i = w[i*head_dim:(i+1)*head_dim,:] #(head_dim, hidden_dim)
                    #print(w_i.shape, linear.in_scale_matrix[i].shape)
                    w_i = torch.matmul(w_i, in_scale_matrix_list[i].to(torch.float32)) #(head_dim, hidden_dim)
                    w_list.append(w_i)
                w = torch.cat(w_list, dim=0) # (num_kv_heads * head_dim, hidden_dim)
                #w_ori = w
                #print('shape', w.shape)
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")

                #w_target_ori_lowrank = torch.matmul(w_target_lowrank, in_scale_inv_ori)

                if args.v_aware:
                    activation_value = torch.mm(linear.in_hidden_matrix, w_ori.T)
                    in_scale_inv = matrix_solve(linear.in_hidden_matrix, w, activation_value)
                else:
                    in_scale_inv = torch.linalg.lstsq(w, w_ori).solution
                    #方案一
                    #in_scale_inv = torch.linalg.lstsq(w, w_target_ori_lowrank).solution
                    #方案二
                    #in_scale_inv = torch.linalg.lstsq(V_ori_lowrank, V_target_ori_lowrank).solution


        if (torch.isnan(w).any()):
            print("nan")
        if (torch.isinf(w).any()):
            print("inf")
        U_, S_, V_ = torch.linalg.svd(w, full_matrices=False)
        U = U_[:,:rank]
        S = S_[:rank]
        V = V_[:rank,:].t() # S_inv * V * sigma * U^T
        if act_aware:
            if hasattr(linear, "in_scale_matrix"):
                V = torch.matmul(in_scale_inv.to(torch.float32).t(), V)

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        new_linear = SVDLinear(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V.to(linear.weight.dtype), bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        new_linear.to(device=linear.weight.device)
        new_linear.truncation_rank = rank

        return new_linear
    

    @staticmethod
    def from_linear_matrix_v5(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        alphaq=1,
        sigma_fuse="None",
        rank_align=1,
        middle=False,
        name=None,
        whiten=False,
    ):
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        assert ic_split == 1 or oc_split == 1
        # compressed_params = int(n_params * param_ratio)
        #rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(rank // 8)
        print(rank)

        w = linear.weight.data.float()
        w = w.to(torch.float32)
        if act_aware:
            in_scale_matrix = 1  # avoid zero division
            out_scale_matrix = 1
            scale_exp_in = None
            scale_exp_out = None
            if hasattr(linear, "in_scale_matrix"):
                in_scale_matrix *= linear.in_scale_matrix
                w = torch.matmul(w, in_scale_matrix.to(torch.float32))
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")
                if whiten == False:
                    in_scale_inv = torch.linalg.inv(in_scale_matrix.transpose(-1, -2).to(torch.float32))
                else:
                    in_scale_inv = linear.in_scale_inv_matrix.to(torch.float32).t()
            if hasattr(linear, "out_scale_matrix"):
                print("do_out_scale")
                out_scale_matrix *= linear.out_scale_matrix.view(-1,1)
                w *= out_scale_matrix
        
        w = w.view(8, 128, 4096).transpose(0, 1).reshape(128, 8 * 4096)
        


        Us = []
        Ss = []
        Vs = []
        if (torch.isnan(w).any()):
            print("nan")
        if (torch.isinf(w).any()):
            print("inf")
        U_, S_, V_ = torch.linalg.svd(w, full_matrices=False)
        U = U_[:,:rank] #(128, rank)
        S = S_[:rank]
        V = V_[:rank,:].t()
        if act_aware:
            if hasattr(linear, "in_scale_matrix"):
                for i in range(8):
                    V[4096 * i:4096 * (i+1)] = torch.mm(torch.matmul(in_scale_inv.to(torch.float32), V[4096 * i:4096 * (i+1)]), torch.diag(S.sqrt()))
                V = V.view(8, 4096, rank).transpose(0,1).reshape(4096, 8 * rank) #(4069, 8 * rank)
                tmp = torch.zeros(128 * 8, 8*rank).to(U.device).to(U.dtype)
                for i in range(8):
                    tmp[128 * i:128 * (i+1), rank * i:rank * (i+1)] = torch.mm(U, torch.diag(S.sqrt()))
                U = tmp
                #U = torch.mm(U, torch.diag(S.sqrt())).repeat(1, 8) #(128, 8 * rank)
            if hasattr(linear, "out_scale_matrix"):
                U /= out_scale_matrix
        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        new_linear = SVDLinear(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V.to(linear.weight.dtype), bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        new_linear.to(device=linear.weight.device)
        new_linear.truncation_rank = rank

        return new_linear
    
    def quantize_latent(self, low_rank_latent):
        assert self.quantizer is not None
        fake_quant_latent = self.quantizer(low_rank_latent)
        return fake_quant_latent

    def forward(self, inp):
        #if inp.size(1) <= 4:
        #    return self.WLinear(inp)
        #else:
        #    inp1 = inp[:,:4]
        #    inp2 = inp[:,4:]
        #y1 = self.WLinear(inp1)
        #y2 = self.BLinear(inp2)
        #y2 = self.ALinear(y2 * self.gate)
        #y = torch.cat([y1, y2],dim = 1)
        # compute USV^Tx + b

        #y = self.BLinear(inp)
        #y = y[:, :self.truncation_rank]
        #weight_A_partial = self.ALinear.weight.data[:, :self.truncation_rank]
        #y = torch.matmul(y, weight_A_partial.t())
        #y = self.ALinear(y)
        #return y

        y = self.BLinear(inp)
        #y = self.ALinear(y * self.gate)
        if self.quant:
            y = self.quantize_latent(y)
        y = self.ALinear(y)
        return y



class SVDLinear_v2(nn.Module):
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV", rank = None) -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)
        self.gate = nn.Parameter(torch.ones(1, U.size(1)).to(self.ALinear.weight.data.dtype))

        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)
        self.truncation_rank = S.size(0)
        if sigma_fuse == "UV":
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == "U":
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == "V":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()
        elif sigma_fuse == "None":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().contiguous()


    def update_scaling_matrix(
        self,
        in_scale_matrix_list,
        num_kv_heads = 8,
        hidden_dim = 4096,
        head_dim = 128,    
    ):
        rank = self.truncation_rank
        V_t = self.BLinear.weight.data
        U = self.ALinear.weight.data
        ori_linear_weight = torch.matmul(U, V_t)
        w_list = []
        for i in range(len(in_scale_matrix_list)):
            in_scale_matrix_i = in_scale_matrix_list[i].to(ori_linear_weight.device)
            w_i = ori_linear_weight[i*head_dim:(i+1)*head_dim, :]
            w_i = torch.mm(w_i, in_scale_matrix_i.to(torch.float32))
            w_list.append(w_i)
        w = torch.cat(w_list, dim = 0)
        U_tmp, S_tmp, V_tmp = torch.linalg.svd(w, full_matrices=False)
        U_tmp = U_tmp[:,:rank]
        S_tmp = S_tmp[:rank]
        V_tmp = V_tmp[:rank,:]
        w_lowrank = torch.matmul(U_tmp, torch.diag(S_tmp)).mm(V_tmp) # (num_kv_heads * head_dim, hidden_dim)

        w_target_ori_lowrank = []
        for i in range(num_kv_heads):
            in_scale_inv_i = torch.linalg.inv(in_scale_matrix_list[i].to(torch.float32).to(w.device))
            w_lowrank_i = w_lowrank[i*head_dim:(i+1)*head_dim,:]
            w_target_ori_lowrank.append(torch.mm(w_lowrank_i, in_scale_inv_i))
        w_target_ori_lowrank = torch.cat(w_target_ori_lowrank, dim=0)
        
        in_scale_inv = torch.linalg.lstsq(w, w_target_ori_lowrank).solution
        P, S, Q = torch.linalg.svd(w, full_matrices=False) 
        Q = Q.t()
        Q = torch.matmul(in_scale_inv.to(torch.float32).t(), Q)
        self.ALinear.weight.data = P.mul(S.sqrt()).contiguous()
        self.BLinear.weight.data = Q.t().mul(S.sqrt().view(-1, 1)).contiguous()
        
    
    @staticmethod
    def from_linear_matrix_v2(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        num_kv_heads=8,
        hidden_dim=4096,
        head_dim = 128,
        sigma_fuse="UV",
        rank_align=1,
        middle=False,
        name=None,
        args=None,
    ):
        
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        assert ic_split == 1 or oc_split == 1
        # compressed_params = int(n_params * param_ratio)
        #rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(rank)
        print(rank)

        w = linear.weight.data.float()
        w = w.to(torch.float32)
        if act_aware:
            in_scale_matrix_list = [1 for _ in range(num_kv_heads)]  # avoid zero division
            out_scale_matrix = 1
            scale_exp_in = None
            scale_exp_out = None
            w_list = []
            w_ori = w
            if hasattr(linear, "in_scale_matrix"):
                #mean_in_scale_maxtrix = torch.mean(torch.stack(linear.in_scale_matrix), dim = 0)
                #w_target = torch.matmul(w_ori, mean_in_scale_maxtrix.to(w_ori.device).to(torch.float32))

                for i in range(num_kv_heads):
                    in_scale_matrix_list[i] = linear.in_scale_matrix[i].to(w_ori.device) # (hidden_dim, hidden_dim)
                    w_i = w[i*head_dim:(i+1)*head_dim,:] #(head_dim, hidden_dim)
                    #print(w_i.shape, linear.in_scale_matrix[i].shape)
                    w_i = torch.matmul(w_i, in_scale_matrix_list[i].to(torch.float32)) #(head_dim, hidden_dim)
                    w_list.append(w_i)
                w = torch.cat(w_list, dim=0) # (num_kv_heads * head_dim, hidden_dim)
                #w_ori = w
                #print('shape', w.shape)
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")

                U_tmp, S_tmp, V_tmp = torch.linalg.svd(w, full_matrices=False)
                U_tmp = U_tmp[:,:rank]
                S_tmp = S_tmp[:rank]
                V_tmp = V_tmp[:rank,:]
                w_lowrank = torch.matmul(U_tmp, torch.diag(S_tmp)).mm(V_tmp) # (num_kv_heads * head_dim, hidden_dim)

                '''U_tmp_ori, S_tmp_ori, V_tmp_ori = torch.linalg.svd(w_ori, full_matrices=False)
                U_tmp_ori = U_tmp_ori[:,:rank]
                S_tmp_ori = S_tmp_ori[:rank]
                V_tmp_ori = V_tmp_ori[:rank,:]
                w_ori_lowrank = torch.matmul(U_tmp_ori, torch.diag(S_tmp_ori)).mm(V_tmp_ori) # (num_kv_heads * head_dim, hidden_dim)
                '''
                '''U_tmp_ori, S_tmp_ori, V_tmp_ori = torch.linalg.svd(w_target, full_matrices=False)
                U_tmp_ori = U_tmp_ori[:,:rank]
                S_tmp_ori = S_tmp_ori[:rank]
                V_tmp_ori = V_tmp_ori[:rank,:]
                w_target_lowrank = torch.matmul(U_tmp_ori, torch.diag(S_tmp_ori)).mm(V_tmp_ori)'''
                #方案一
                w_target_ori_lowrank = []
                for i in range(num_kv_heads):
                    in_scale_inv_i = torch.linalg.inv(in_scale_matrix_list[i].to(torch.float32).to(w.device))
                    w_lowrank_i = w_lowrank[i*head_dim:(i+1)*head_dim,:]
                    w_target_ori_lowrank.append(torch.mm(w_lowrank_i, in_scale_inv_i))
                w_target_ori_lowrank = torch.cat(w_target_ori_lowrank, dim=0) #(num_kv_heads * head_dim, hidden_dim)
                    
                #方案二
                V_target_ori_lowrank = []
                for i in range(num_kv_heads):
                    in_scale_inv_i = torch.linalg.inv(in_scale_matrix_list[i].to(torch.float32).to(w.device))
                    V_target_ori_lowrank.append(torch.mm(V_tmp, in_scale_inv_i))
                V_target_ori_lowrank = torch.cat(V_target_ori_lowrank, dim=0) #(num_kv_heads * rank, hidden_dim)
                #V_ori_lowrank = torch.cat([V_tmp for _ in range(num_kv_heads)], dim=0) #(num_kv_heads * rank, hidden_dim)

                    
                #in_scale_inv_ori = torch.linalg.inv(mean_in_scale_maxtrix.to(torch.float32).to(w.device))
                #w_target_ori_lowrank = torch.matmul(w_target_lowrank, in_scale_inv_ori)

                if args.v_aware:
                    activation_value = torch.mm(linear.in_hidden_matrix, w_ori.T)
                    in_scale_inv = matrix_solve(linear.in_hidden_matrix, w, activation_value)
                else:
                    #in_scale_inv = torch.linalg.lstsq(w, w_ori).solution
                    #方案一
                    in_scale_inv = torch.linalg.lstsq(w, w_target_ori_lowrank).solution
                    #方案二
                    #in_scale_inv = torch.linalg.lstsq(V_ori_lowrank, V_target_ori_lowrank).solution


        if (torch.isnan(w).any()):
            print("nan")
        if (torch.isinf(w).any()):
            print("inf")


        U_, S_, V_ = torch.linalg.svd(w, full_matrices=False) 
        #U = U_ #这里改成保存满秩的USV，为了后续推理时更新Scaling矩阵
        #S = S_
        #V = V_.t()
        U = U_[:,:rank]
        S = S_[:rank]
        V = V_[:rank,:].t() # S_inv * V * sigma * U^T
        if act_aware:
            if hasattr(linear, "in_scale_matrix"):
                V = torch.matmul(in_scale_inv.to(torch.float32).t(), V)

        #U_, S_, V_ = torch.linalg.svd(w_target_ori_lowrank, full_matrices=False)
        #U = U_#[:,:rank]
        #S = S_#[:rank]
        #V = V_.t()#[:rank,:].t() # S_inv * V * sigma * U^T

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        new_linear = SVDLinear(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V.to(linear.weight.dtype), bias, sigma_fuse)
        #new_linear = SVDLinear_v2(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V_target_ori_lowrank.t().to(linear.weight.dtype), bias, sigma_fuse, rank=rank)
        new_linear.to(linear.weight.dtype)
        new_linear.to(device=linear.weight.device)
        new_linear.truncation_rank = rank

        return new_linear
    
    @staticmethod
    def from_linear_matrix_v2_for_quant(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        num_kv_heads=8,
        hidden_dim=4096,
        head_dim = 128,
        sigma_fuse="UV",
        rank_align=1,
        middle=False,
        name=None,
        args=None,
    ):
        
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        assert ic_split == 1 or oc_split == 1
        # compressed_params = int(n_params * param_ratio)
        #rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(rank)
        print(rank)

        w = linear.weight.data.float()
        w = w.to(torch.float32)
        if act_aware:
            in_scale_matrix_list = [1 for _ in range(num_kv_heads)]  # avoid zero division
            out_scale_matrix = 1
            scale_exp_in = None
            scale_exp_out = None
            w_list = []
            w_ori = w
            if hasattr(linear, "in_scale_matrix"):
                #mean_in_scale_maxtrix = torch.mean(torch.stack(linear.in_scale_matrix), dim = 0)
                #w_target = torch.matmul(w_ori, mean_in_scale_maxtrix.to(w_ori.device).to(torch.float32))

                for i in range(num_kv_heads):
                    in_scale_matrix_list[i] = linear.in_scale_matrix[i].to(w_ori.device) # (hidden_dim, hidden_dim)
                    w_i = w[i*head_dim:(i+1)*head_dim,:] #(head_dim, hidden_dim)
                    #print(w_i.shape, linear.in_scale_matrix[i].shape)
                    w_i = torch.matmul(w_i, in_scale_matrix_list[i].to(torch.float32)) #(head_dim, hidden_dim)
                    w_list.append(w_i)
                w = torch.cat(w_list, dim=0) # (num_kv_heads * head_dim, hidden_dim)
                #w_ori = w
                #print('shape', w.shape)
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")

                U_tmp, S_tmp, V_tmp = torch.linalg.svd(w, full_matrices=False)
                U_tmp = U_tmp[:,:rank]
                S_tmp = S_tmp[:rank]
                V_tmp = V_tmp[:rank,:]
                w_lowrank = torch.matmul(U_tmp, torch.diag(S_tmp)).mm(V_tmp) # (num_kv_heads * head_dim, hidden_dim)

                '''U_tmp_ori, S_tmp_ori, V_tmp_ori = torch.linalg.svd(w_ori, full_matrices=False)
                U_tmp_ori = U_tmp_ori[:,:rank]
                S_tmp_ori = S_tmp_ori[:rank]
                V_tmp_ori = V_tmp_ori[:rank,:]
                w_ori_lowrank = torch.matmul(U_tmp_ori, torch.diag(S_tmp_ori)).mm(V_tmp_ori) # (num_kv_heads * head_dim, hidden_dim)
                '''
                '''U_tmp_ori, S_tmp_ori, V_tmp_ori = torch.linalg.svd(w_target, full_matrices=False)
                U_tmp_ori = U_tmp_ori[:,:rank]
                S_tmp_ori = S_tmp_ori[:rank]
                V_tmp_ori = V_tmp_ori[:rank,:]
                w_target_lowrank = torch.matmul(U_tmp_ori, torch.diag(S_tmp_ori)).mm(V_tmp_ori)'''
                #方案一
                w_target_ori_lowrank = []
                for i in range(num_kv_heads):
                    in_scale_inv_i = torch.linalg.inv(in_scale_matrix_list[i].to(torch.float32).to(w.device))
                    w_lowrank_i = w_lowrank[i*head_dim:(i+1)*head_dim,:]
                    w_target_ori_lowrank.append(torch.mm(w_lowrank_i, in_scale_inv_i))
                w_target_ori_lowrank = torch.cat(w_target_ori_lowrank, dim=0) #(num_kv_heads * head_dim, hidden_dim)
                    
                #方案二
                V_target_ori_lowrank = []
                for i in range(num_kv_heads):
                    in_scale_inv_i = torch.linalg.inv(in_scale_matrix_list[i].to(torch.float32).to(w.device))
                    V_target_ori_lowrank.append(torch.mm(V_tmp, in_scale_inv_i))
                V_target_ori_lowrank = torch.cat(V_target_ori_lowrank, dim=0) #(num_kv_heads * rank, hidden_dim)
                #V_ori_lowrank = torch.cat([V_tmp for _ in range(num_kv_heads)], dim=0) #(num_kv_heads * rank, hidden_dim)

                    
                #in_scale_inv_ori = torch.linalg.inv(mean_in_scale_maxtrix.to(torch.float32).to(w.device))
                #w_target_ori_lowrank = torch.matmul(w_target_lowrank, in_scale_inv_ori)

                if args.v_aware:
                    activation_value = torch.mm(linear.in_hidden_matrix, w_ori.T)
                    in_scale_inv = matrix_solve(linear.in_hidden_matrix, w, activation_value)
                else:
                    in_scale_inv = torch.linalg.lstsq(w, w_ori).solution
                    #方案一
                    #in_scale_inv = torch.linalg.lstsq(w, w_target_ori_lowrank).solution
                    #方案二
                    #in_scale_inv = torch.linalg.lstsq(V_ori_lowrank, V_target_ori_lowrank).solution


        if (torch.isnan(w).any()):
            print("nan")
        if (torch.isinf(w).any()):
            print("inf")


        U_, S_, V_ = torch.linalg.svd(w, full_matrices=False) 
        #U = U_ #这里改成保存满秩的USV，为了后续推理时更新Scaling矩阵
        #S = S_
        #V = V_.t()
        U = U_
        S = S_
        V = V_.t() # S_inv * V * sigma * U^T
        if act_aware:
            if hasattr(linear, "in_scale_matrix"):
                V = torch.matmul(in_scale_inv.to(torch.float32).t(), V)

        #U_, S_, V_ = torch.linalg.svd(w_target_ori_lowrank, full_matrices=False)
        #U = U_#[:,:rank]
        #S = S_#[:rank]
        #V = V_.t()#[:rank,:].t() # S_inv * V * sigma * U^T

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        new_linear = SVDLinear(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V.to(linear.weight.dtype), bias, sigma_fuse)
        #new_linear = SVDLinear_v2(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V_target_ori_lowrank.t().to(linear.weight.dtype), bias, sigma_fuse, rank=rank)
        new_linear.to(linear.weight.dtype)
        new_linear.to(device=linear.weight.device)
        new_linear.truncation_rank = rank

        return new_linear


    @staticmethod
    def from_linear_matrix_v2_for_quant_key(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        num_kv_heads=8,
        hidden_dim=4096,
        head_dim = 128,
        sigma_fuse="UV",
        rank_align=1,
        middle=False,
        name=None,
        args=None,
    ):
        
        
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        assert ic_split == 1 or oc_split == 1
        # compressed_params = int(n_params * param_ratio)
        #rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(rank)
        print(rank)

        w = linear.weight.data.float()
        w = w.to(torch.float32)
        w_ori = w
        if act_aware:
            in_scale_matrix = 1  # avoid zero division
            out_scale_matrix = 1
            scale_exp_in = None
            scale_exp_out = None
            if hasattr(linear, "in_scale_matrix"):
                in_scale_matrix *= linear.in_scale_matrix
                w = torch.matmul(w, in_scale_matrix.to(torch.float32))
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")
                
                in_scale_inv = torch.linalg.inv(in_scale_matrix.transpose(-1, -2).to(torch.float32))
                
            if hasattr(linear, "out_scale_matrix"):
                print("do_out_scale")
                out_scale_matrix *= linear.out_scale_matrix.view(-1,1)
                w *= out_scale_matrix


        Us = []
        Ss = []
        Vs = []
        if (torch.isnan(w).any()):
            print("nan")
        if (torch.isinf(w).any()):
            print("inf")
        U_, S_, V_ = torch.linalg.svd(w, full_matrices=False)
        U = U_#[:,:rank]
        S = S_#[:rank]
        V = V_.t()#[:rank,:].t()
        if act_aware:
            if hasattr(linear, "in_scale_matrix"):
                V = torch.matmul(in_scale_inv.to(torch.float32), V)
            if hasattr(linear, "out_scale_matrix"):
                U /= out_scale_matrix
        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        new_linear = SVDLinear(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V.to(linear.weight.dtype), bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        new_linear.to(device=linear.weight.device)
        new_linear.truncation_rank = rank

        return new_linear
    
    @staticmethod
    def from_linear_matrix_v2_2(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        num_kv_heads=8,
        hidden_dim=4096,
        head_dim = 128,
        sigma_fuse="None",
        rank_align=1,
        middle=False,
        name=None,
        args=None,
    ):
        
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        assert ic_split == 1 or oc_split == 1
        # compressed_params = int(n_params * param_ratio)
        #rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(rank // 8)
        print(rank)

        w = linear.weight.data.float()
        w = w.to(torch.float32)
        if act_aware:
            in_scale_matrix_list = [1 for _ in range(num_kv_heads)]  # avoid zero division
            out_scale_matrix = 1
            scale_exp_in = None
            scale_exp_out = None
            w_list = []
            w_ori = w
            if hasattr(linear, "in_scale_matrix"):
                #mean_in_scale_maxtrix = torch.mean(torch.stack(linear.in_scale_matrix), dim = 0)
                #w_target = torch.matmul(w_ori, mean_in_scale_maxtrix.to(w_ori.device).to(torch.float32))

                for i in range(num_kv_heads):
                    in_scale_matrix_list[i] = linear.in_scale_matrix[i].to(w_ori.device) # (hidden_dim, hidden_dim)
                    w_i = w[i*head_dim:(i+1)*head_dim,:] #(head_dim, hidden_dim)
                    #print(w_i.shape, linear.in_scale_matrix[i].shape)
                    w_i = torch.matmul(w_i, in_scale_matrix_list[i].to(torch.float32)) #(head_dim, hidden_dim)
                    w_list.append(w_i)
                w = torch.cat(w_list, dim=0) # (num_kv_heads * head_dim, hidden_dim)
                w = w.view(8, 128, 4096).transpose(0,1).reshape(128, 8 * 4096)
                #w_ori = w
                #print('shape', w.shape)
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")

                U_tmp, S_tmp, V_tmp = torch.linalg.svd(w, full_matrices=False)
                U_tmp = U_tmp[:,:rank]
                S_tmp = S_tmp[:rank]
                V_tmp = V_tmp[:rank,:].t()
                #w_lowrank = torch.matmul(U_tmp, torch.diag(S_tmp)).mm(V_tmp) # (head_dim, num kv head * hidden_dim)

                #方案一
                V_target_ori_lowrank = []
                for i in range(num_kv_heads):
                    in_scale_inv_i = torch.linalg.inv(in_scale_matrix_list[i].to(torch.float32).to(w.device)).t()
                    V_lowrank_i = torch.mm(V_tmp[i*hidden_dim:(i+1)*hidden_dim, :], torch.diag(S_tmp.sqrt())) # (4096, rank)
                    V_target_ori_lowrank.append(torch.mm(in_scale_inv_i, V_lowrank_i))
                #w_target_ori_lowrank = torch.cat(w_target_ori_lowrank, dim=0) #(num_kv_heads * head_dim, hidden_dim)
                V_target_ori_lowrank = torch.cat(V_target_ori_lowrank, dim = 1)
                    
                



        if (torch.isnan(w).any()):
            print("nan")
        if (torch.isinf(w).any()):
            print("inf")



        U = U_tmp #torch.mm(U_tmp, torch.diag(S_tmp.sqrt())).repeat(1, 8)
        tmp = torch.zeros(128 * 8, rank * 8).to(U.device).to(U.dtype)
        for i in range(8):
            tmp[i*128:(i+1)*128, i*rank:(i+1)*rank] = torch.mm(U_tmp, torch.diag(S_tmp.sqrt()))
        U = tmp
        S = S_tmp
        V = V_target_ori_lowrank

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        new_linear = SVDLinear_v2(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V.to(linear.weight.dtype), w_ori, bias, sigma_fuse)
        #new_linear = SVDLinear_v2(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V_target_ori_lowrank.t().to(linear.weight.dtype), bias, sigma_fuse, rank=rank)
        new_linear.to(linear.weight.dtype)
        new_linear.to(device=linear.weight.device)
        new_linear.truncation_rank = rank

        return new_linear


    


    def forward(self, inp):
        #if inp.size(1) <= 4:
        #    return self.WLinear(inp)
        #else:
        #    inp1 = inp[:,:4]
        #    inp2 = inp[:,4:]
        #y1 = self.WLinear(inp1)
        #y2 = self.BLinear(inp2)
        #y2 = self.ALinear(y2 * self.gate)
        #y = torch.cat([y1, y2],dim = 1)
        # compute USV^Tx + b


        y = self.BLinear(inp)
        y = y[:, :self.truncation_rank]
        weight_A_partial = self.ALinear.weight[:, :self.truncation_rank]
        y = torch.matmul(y, weight_A_partial.t())
        #y = self.ALinear(y)
        return y
        #y = self.BLinear(inp)
        #y = self.ALinear(y * self.gate)
        #return y
        #if inp.size(1) <= 4:
        #    return self.WLinear(inp)
        #else:
            #inp1 = inp[:,:4]
            #inp2 = inp[:,4:]
        # compute USV^Tx + b
            #y1 = self.WLinear(inp1)
            #y = self.BLinear(inp)
            ###
        '''y_list = [] #y有batch size这个维度
        for i in range(8):
            yi = y[:,:,i*self.truncation_rank:(i+1)*self.truncation_rank]
            y_list.append(yi * self.gate)  
        y = torch.cat(y_list, dim=-1) #(bs, seq_len, num_kv_heads * rank)
        WA = self.ALinear.weight #(num_kv_heads * head_dim, rank)
        out_list = []
        for i in range(8):
            Wi = WA[i*128:(i+1)*128,:] #(head_dim, rank)
            yi = y[:,:,i*self.truncation_rank:(i+1)*self.truncation_rank] #(bs, seq_len, rank)
            outi = torch.matmul(yi, Wi.t()) #(bs, seq_len, head_dim)
            out_list.append(outi)
        return torch.cat(out_list, dim=-1) #(bs, seq_len, num_kv_heads * head_dim)'''

        ###
    

class SVDLinear_v3(nn.Module):
    def __init__(self, U, S, V_list, bias=None, sigma_fuse="UV") -> None:
        super().__init__()
        #self.ALinear = nn.Parameter(torch.zeros(U.size(0), U.size(1)))
        self.ALinear_list = nn.ModuleList([nn.Linear(U.size(1), U.size(0) // len(V_list)) for _ in range(len(V_list))])
        self.gate = nn.Parameter(torch.ones(1, U.size(1)).to(self.ALinear_list[0].weight.data.dtype))

        #if bias is not None:
        #    self.ALinear.bias.data = bias
        self.BLinear_list = nn.ModuleList([nn.Linear(Vi.size(0), Vi.size(1)) for Vi in V_list])
        self.truncation_rank = S.size(0)
        if sigma_fuse == "UV":
            U2 = U.mul(S.sqrt()).contiguous()
            for i in range(len(self.BLinear_list)):
                self.ALinear_list[i].weight.data = U2[i*U.size(0)//len(V_list):(i+1)*U.size(0)//len(V_list), :].mul(S.sqrt()).contiguous()
                self.BLinear_list[i].weight.data = V_list[i].t().mul(S.sqrt().view(-1, 1)).contiguous()
        '''elif sigma_fuse == "U":
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == "V":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()'''
    @staticmethod
    def from_linear_matrix_v3(
        linear: nn.Linear,
        rank: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        num_kv_heads=8,
        hidden_dim=4096,
        head_dim = 128,
        sigma_fuse="UV",
        rank_align=1,
        middle=False,
        name=None,
    ):
        
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()

        assert ic_split == 1 or oc_split == 1
        # compressed_params = int(n_params * param_ratio)
        #rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(rank)
        print(rank)

        w = linear.weight.data.float()
        w = w.to(torch.float32)
        if act_aware:
            in_scale_matrix_list = [1 for _ in range(num_kv_heads)]  # avoid zero division
            out_scale_matrix = 1
            scale_exp_in = None
            scale_exp_out = None
            w_list = []
            w_ori = w
            in_scale_inv_list = []
            if hasattr(linear, "in_scale_matrix"):
                for i in range(num_kv_heads):
                    in_scale_matrix_list[i] = linear.in_scale_matrix[i].to(w.device) # (hidden_dim, hidden_dim)
                    w_i = w[i*head_dim:(i+1)*head_dim,:] #(head_dim, hidden_dim)
                    print(w_i.shape, linear.in_scale_matrix[i].shape)
                    w_i = torch.matmul(w_i, in_scale_matrix_list[i].to(torch.float32)) #(head_dim, hidden_dim)
                    w_list.append(w_i)
                    in_scale_inv_list.append(torch.linalg.inv(in_scale_matrix_list[i]).to(torch.float32))
                w = torch.cat(w_list, dim=0) # (num_kv_heads * head_dim, hidden_dim)
                print('shape', w.shape)
                if (torch.isnan(w).any()):
                    print("nan")
                if (torch.isinf(w).any()):
                    print("inf")
                U_tmp, S_tmp, V_tmp = torch.linalg.svd(w, full_matrices=False)
                U_tmp = U_tmp[:,:rank]
                S_tmp = S_tmp[:rank]
                V_tmp = V_tmp[:rank,:]
                w = torch.matmul(U_tmp, torch.diag(S_tmp)).mm(V_tmp) # (num_kv_heads * head_dim, hidden_dim)
                #in_scale_inv = torch.linalg.lstsq(w, w_ori).solution



        if (torch.isnan(w).any()):
            print("nan")
        if (torch.isinf(w).any()):
            print("inf")
        V_list = []
        U_, S_, V_ = torch.linalg.svd(w, full_matrices=False)
        U = U_[:,:rank]
        S = S_[:rank]
        V = V_[:rank,:].t() # S_inv * V * sigma * U^T
        
        if act_aware:
            if hasattr(linear, "in_scale_matrix"):
                for i in range(num_kv_heads):
                    Vi = torch.matmul(in_scale_inv_list[i].t(), V)
                    V_list.append(Vi)

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        new_linear = SVDLinear_v3(U.to(linear.weight.dtype), S.to(linear.weight.dtype), V_list, bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        new_linear.to(device=linear.weight.device)
        new_linear.truncation_rank = rank

        return new_linear


    def forward(self, inp):
        # compute USV^Tx + b
        y_list = []
        for i in range(len(self.BLinear_list)):
            y_i = self.BLinear_list[i](inp)
            y_list.append(self.ALinear_list[i](y_i * self.gate))
        #y = self.BLinear(inp)
        #y = self.ALinear(y * self.gate)
        return torch.cat(y_list, dim=1)



class GradSVDLinear(nn.Module):
    def __init__(self, weight, scale, bias, rank) -> None:
        super().__init__()
        self.weight = weight
        self.scale = nn.Parameter(scale)
        self.bias = bias
        self.rank = rank

    @staticmethod
    def from_linear(
        linear: nn.Linear, param_ratio: float, act_aware=False, ic_split=1, oc_split=1, alpha=1, sigma_fuse="UV"
    ):
        if param_ratio >= 1:
            return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        # print("rank", rank)
        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        return GradSVDLinear(w, scaling_diag_matrix, bias, rank)

    def forward(self, inp):
        w = self.weight * self.scale.view(1, -1)
        U, S, V = torch.svd_lowrank(w, q=self.rank)
        new_w = U.mul(S).mm(V.t())
        y = F.linear(inp, new_w, self.bias)
        return y
