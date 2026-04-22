import torch
import argparse
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from model.ours_distill_base_model import Base_Model
import pickle
from torchvision.models import mobilenet_v2

class CIBHash(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')

        # --- [步骤二 手术] ---
        # 移除了所有 get_mask/ 文件的加载 [cite: 28, 31, 32, 33]
        # (此处原有 20+ 行加载 .pkl 和 .txt 文件的代码已被删除)
        # ---------------------
        
        if self.hparams.s_model_name == 'mobilenet_v2':
            self.mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)
            print("use mobilenet_v2 as backbone")

        if self.hparams.s_model_name == 'resnet18':
            self.resnet = torchvision.models.resnet18(pretrained=True)
            print("use resnet18 as backbone")
            block_num = 1
        if self.hparams.s_model_name == 'resnet34':
            self.resnet = torchvision.models.resnet34(pretrained=True)
            print("use resnet34 as backbone")
            block_num = 1
        if self.hparams.s_model_name == 'resnet50':
            self.resnet = torchvision.models.resnet50(pretrained=True)
            print("use resnet50 as backbone")
            block_num = 4
        if self.hparams.s_model_name == 'resnet101':
            self.resnet = torchvision.models.resnet101(pretrained=True)
            print("use resnet101 as backbone")
            block_num = 4
        if self.hparams.s_model_name == 'resnet152':
            self.resnet = torchvision.models.resnet152(pretrained=True)
            print("use resnet152 as backbone")
            block_num = 4
        
        if self.hparams.s_model_name == 'efficientnet_b0':
            self.efficient_net = torchvision.models.efficientnet_b0(pretrained=True)
            print("use efficientnet_b0 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b1':
            self.efficient_net = torchvision.models.efficientnet_b1(pretrained=True)
            print("use efficientnet_b1 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b2':
            self.efficient_net = torchvision.models.efficientnet_b2(pretrained=True)
            print("use efficientnet_b2 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b3':
            self.efficient_net = torchvision.models.efficientnet_b3(pretrained=True)
            print("use efficientnet_b3 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b4':
            self.efficient_net = torchvision.models.efficientnet_b4(pretrained=True)
            print("use efficientnet_b4 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b5':
            self.efficient_net = torchvision.models.efficientnet_b5(pretrained=True)
            print("use efficientnet_b5 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b6':
            self.efficient_net = torchvision.models.efficientnet_b6(pretrained=True)
            print("use efficientnet_b6 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b7':
            self.efficient_net = torchvision.models.efficientnet_b7(pretrained=True)
            print("use efficientnet_b7 as backbone")

        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc = nn.Linear(512 * block_num, self.hparams.encode_length)
        
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            for param in self.efficient_net.parameters():
                param.requires_grad = False
            self.fc = nn.Sequential(nn.Linear(1000, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, self.hparams.encode_length),
                                   )
        if self.hparams.s_model_name in ('mobilenet_v2',):
            for param in self.mobilenet_v2.parameters():
                param.requires_grad = False
            self.fc = nn.Sequential(nn.Linear(1000, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, self.hparams.encode_length),
                                   )

        self.criterion = NtXentLoss(self.hparams.batch_size, self.hparams.temperature)
        self.criterion_distill = BRCDLoss(self.hparams.batch_size, self.hparams.temperature)
    
    def forward(self, raw_imgi, raw_imgj, idxs, device):
        # raw_imgi 是 view_1 (弱增强) [cite: 21]
        # raw_imgj 是 view_2 (强增强) [cite: 21]
        
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            imgi = self.resnet(raw_imgi)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgi = self.efficient_net(raw_imgi)
            imgi = self.fc(imgi)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            imgi = self.mobilenet_v2(raw_imgi)
            imgi = self.fc(imgi)
        prob_i = torch.sigmoid(imgi)
        z_i = hash_layer(prob_i - 0.5) # 学生 (view 1)

        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            imgj = self.resnet(raw_imgj)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgj = self.efficient_net(raw_imgj)
            imgj = self.fc(imgj)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            imgj = self.mobilenet_v2(raw_imgj)
            imgj = self.fc(imgj)
        prob_j = torch.sigmoid(imgj)
        z_j = hash_layer(prob_j - 0.5) # 学生 (view 2)

        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
        
        # 这是学生模型 view_1 和 view_2 之间的对比损失
        contra_loss = self.criterion(z_i, z_j, device)
        
        with torch.no_grad():
            t_z_i = self.t_model.encode_discrete(raw_imgi) # 教师 (view 1)
            t_z_j = self.t_model.encode_discrete(raw_imgj) # 教师 (view 2)

        # --- [步骤三 手术] ---
        # 移除了所有 'false_neg', 'revise_distance', 'false_pos' 逻辑
        # (此处原有 40+ 行有害的语义逻辑已被删除)
        
        # [核心修改] 
        distll_loss = self.criterion_distill(
            z_i,      # 学生 view 1 (h_s_i)
            t_z_i,    # 教师 view 1 (h_t_i)
            t_z_j,    # 教师 view 2 (h_t_i')
            self.hparams.alpha, # 固定的 alpha 超参数
            device
        )
        # --- [手术结束] ---

        loss = contra_loss + self.hparams.weight * kl_loss + distll_loss

        return {'loss': loss, 'contra_loss': contra_loss, 'kl_loss': kl_loss}
    
    def encode_discrete(self, x):
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            x = self.resnet(x)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            x = self.efficient_net(x)
            x = self.fc(x)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            x = self.mobilenet_v2(x)
            x = self.fc(x)
        prob = torch.sigmoid(x)
        z = hash_layer(prob - 0.5)

        return z

    def compute_kl(self, prob, prob_v):
        prob_v = prob_v.detach()
        kl = prob * (torch.log(prob + 1e-8) - torch.log(prob_v + 1e-8)) + (1 - prob) * (torch.log(1 - prob + 1e-8 ) - torch.log(1 - prob_v + 1e-8))
        kl = torch.mean(torch.sum(kl, axis = 1))
        return kl

    def configure_optimizers(self):
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            return torch.optim.Adam([{'params': self.resnet.fc.parameters()}], lr = self.hparams.lr)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            return torch.optim.Adam([{'params': self.fc.parameters()}], lr = self.hparams.lr)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            return torch.optim.Adam([{'params': self.fc.parameters()}], lr = self.hparams.lr)

    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'temperature': [0.2, 0.3, 0.4],
            'weight': [0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument("-t", "--temperature", default = 0.3, type = float,
                            help = "Temperature [%(default)d]",)
        parser.add_argument('-w',"--weight", default = 0.001, type=float,
                            help='weight of I(x,z) [%(default)f]')
        parser.add_argument("--l2_weight", default = 1, type=float)
        parser.add_argument("--l1_weight", default = 1, type=float)
        parser.add_argument("--kl_distill_weight", default = 1, type=float)
        return parser


class hash(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def hash_layer(input):
    return hash.apply(input)


class BRCDLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(BRCDLoss, self).__init__()
        self.temperature = temperature
        self.similarityF = nn.CosineSimilarity(dim = 2)
        self.similarityA = nn.CosineSimilarity(dim = 1)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    # --- [步骤三 手术] ---
    def forward(self, z_i, t_z_i, t_z_j, alpha, device): # <-- 1. 修改函数签名 [cite: 36]
        """
        z_i: anchor image (view 1) from student model
        t_z_i: anchor image (view 1) from teacher model
        t_z_j: augment image (view 2) from teacher model
        alpha: 固定的超参数 [cite: 44]
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        
        # 2. 移除 z_i_reg (比特掩码) 逻辑 [cite: 39-41]
        z = torch.cat((z_i, t_z_j), dim=0) 

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # sim_anchor = phi(z_i, t_z_i) (个体空间知识) [cite: 48]
        sim_anchor = self.similarityA(z_i, t_z_i) / self.temperature
        sim_anchor = torch.cat((sim_anchor, sim_anchor), dim=0)

        # sim_i_j = phi(z_i, t_z_j) (新含义: 鲁棒性) [cite: 49]
        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        
        positive_from_aug = torch.cat((sim_i_j, sim_j_i), dim=0)
        
        # 3. 使用固定的 'alpha' 替换 'alpha_vec' [cite: 44]
        positive_samples = alpha * sim_anchor + (1 - alpha) * positive_from_aug
        positive_samples = positive_samples.view(N, 1)
        
        # 4. 确保不使用 'fn_matrix' (假负样本过滤) [cite: 45, 46]
        negative_samples = sim[mask].view(N, -1)
        
        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class NtXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NtXentLoss, self).__init__()
        self.temperature = temperature
        self.similarityF = nn.CosineSimilarity(dim = 2)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def forward(self, z_i, z_j, device):
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)
        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss