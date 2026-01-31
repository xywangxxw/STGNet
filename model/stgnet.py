import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch

from losses import CELoss
from losses import Huber2DLoss
from metrics import BrierMinFDE
from metrics import MR
from metrics import MinADE
from metrics import MinFDE
from modules import MapEncoder
from modules import SceneEncoder
from modules import TrajDecoder

class STGNet(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 pos_duration: int,
                 pred_duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 dynbr_psi: float,
                 num_visible_steps: int,
                 num_modes: int,
                 num_attn_layers: int,
                 num_hops: int,
                 num_heads: int,
                 dropout: float,
                 lr: float,
                 weight_decay: float,
                 warmup_epochs: int,
                 T_max: int,
                 **kwargs) -> None:
        super(STGNet, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pos_duration = pos_duration
        self.pred_duration = pred_duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_visible_steps = num_visible_steps
        self.num_modes = num_modes
        self.num_attn_layers = num_attn_layers
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max

        self.NewBackbone = SceneEncoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pos_duration=pos_duration,
            a2a_radius=a2a_radius,
            l2a_radius=l2a_radius,
            num_heads=num_heads,
            num_layers=num_attn_layers,
            dropout=dropout
        )
        self.NewDecoder = TrajDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            pos_duration=pos_duration,
            a2a_radius=a2a_radius,
            l2a_radius=l2a_radius,
            dynbr_psi=dynbr_psi,
            num_attn_layers=num_attn_layers,
            num_modes=num_modes,
            num_heads=num_heads,
            dropout=dropout)

        self.MapEncoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )

        self.reg_loss = Huber2DLoss()
        self.prob_loss = CELoss()

        self.brier_minFDE = BrierMinFDE()
        self.minADE = MinADE()
        self.minFDE = MinFDE()
        self.MR = MR()

        self.cnt = 0
        self.test_traj_output = dict()
        self.test_prob_output = dict()

    def forward(self, 
                data: Batch):
        lane_embs = self.MapEncoder(data=data)
        scene_embs = self.NewBackbone(data, lane_embs)
        pred = self.NewDecoder(data, scene_embs)
        return pred

    def training_step(self,data,batch_idx):
        pred = self(data)
        traj_propose = pred['traj_propose']
        traj_refine = pred['traj_refine']
        prob = pred['prob']

        reg_mask = data['agent']['visible_mask'][:, self.num_historical_steps:]
        current_valid_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1].unsqueeze(
            -1).repeat_interleave(self.num_future_steps, -1)
        reg_mask = torch.where(current_valid_mask, reg_mask,
                               torch.zeros(size=(current_valid_mask.shape[0], self.num_future_steps),
                                           dtype=torch.bool).to(reg_mask.device))
        cls_mask = reg_mask[:, -1]

        gt = data['agent']['target']
        l2_norm = (torch.norm(traj_propose - gt.unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_best_refine = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        label = best_mode[cls_mask]
        prob = prob[cls_mask]

        reg_loss_propose = self.reg_loss(traj_best_propose[reg_mask], gt[reg_mask])
        reg_loss_refine = self.reg_loss(traj_best_refine[reg_mask], gt[reg_mask])
        prob_loss = self.prob_loss(prob, label)
        loss = reg_loss_propose + reg_loss_refine + prob_loss
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_prob_loss', prob_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return loss

    def validation_step(self,data,batch_idx):
        pred = self(data)
        traj_propose = pred['traj_propose']
        traj_refine = pred['traj_refine']
        prob = pred['prob']

        reg_mask = data['agent']['visible_mask'][:, self.num_historical_steps:]
        current_valid_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1].unsqueeze(-1).repeat_interleave(self.num_future_steps, -1)
        reg_mask = torch.where(current_valid_mask, reg_mask, torch.zeros(size=(current_valid_mask.shape[0], self.num_future_steps), dtype=torch.bool).to(reg_mask.device))
        cls_mask = reg_mask[:, -1]

        gt = data['agent']['target']
        l2_nrom = (torch.norm(traj_propose - gt.unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_nrom.argmin(dim=-1)
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_best_refine = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        label = best_mode[cls_mask]
        prob = prob[cls_mask]

        reg_loss_propose = self.reg_loss(traj_best_propose[reg_mask], gt[reg_mask])
        reg_loss_refine = self.reg_loss(traj_best_refine[reg_mask], gt[reg_mask])
        prob_loss = self.prob_loss(prob, label)
        loss = reg_loss_propose + reg_loss_refine + prob_loss
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_prob_loss', prob_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        pred_mask = data['agent']['agent_index'] + data['agent']['ptr'][:-1]
        traj_agent = traj_refine[pred_mask]
        gt_agent = gt[pred_mask]

        fde_agent = torch.norm(traj_agent[:, :, -1] - gt_agent[:, -1].unsqueeze(1), p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=-1)
        traj_best_agent = traj_agent[torch.arange(traj_agent.size(0)), best_mode_agent]
        prob_agent = pred['prob'][pred_mask]
        prob_best_agent = prob_agent[torch.arange(traj_agent.size(0)), best_mode_agent]

        self.brier_minFDE.update(traj_best_agent, gt_agent, prob_best_agent)
        self.minADE.update(traj_best_agent, gt_agent)
        self.minFDE.update(traj_best_agent, gt_agent)
        self.MR.update(traj_best_agent, gt_agent)
        self.log('val_Brier', self.brier_minFDE, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=gt_agent.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_agent.size(0))
        self.log('val_minMR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_agent.size(0))

    def configure_optimizers(self):
        import math
        import torch
        import torch.nn as nn

        whitelist_weight_modules = (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.LayerNorm, nn.Embedding
        )

        param_dict = dict(self.named_parameters())

        decay = set()
        no_decay = set()

        module_map = {}
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                module_map[full_name] = module

        for name, param in param_dict.items():
            if not param.requires_grad:
                continue

            if name.endswith(".bias"):
                no_decay.add(name)
                continue

            module = module_map.get(name, None)

            if module is None:
                no_decay.add(name)
            elif isinstance(module, whitelist_weight_modules):
                decay.add(name)
            elif isinstance(module, blacklist_weight_modules):
                no_decay.add(name)
            else:
                decay.add(name)

        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, f"Parameters in both decay and no_decay: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Unclassified parameters: {param_dict.keys() - union_params}"

        optim_groups = [
            {
                "params": [param_dict[n] for n in sorted(decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[n] for n in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (
                    1.0 + math.cos(
                math.pi * (epoch - warmup_epochs + 1)
                / (T_max - warmup_epochs + 1)
            )
            )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, warmup_cosine_annealing_schedule
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HPNet')
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_historical_steps', type=int, default=20)
        parser.add_argument('--num_future_steps', type=int, default=30)
        parser.add_argument('--pos_duration', type=int, default=20)
        parser.add_argument('--pred_duration', type=int, default=20)
        parser.add_argument('--a2a_radius', type=float, default=50)
        parser.add_argument('--l2a_radius', type=float, default=50)
        parser.add_argument('--dynbr_psi', type=float, default=1.5)
        parser.add_argument('--num_visible_steps', type=int, default=2)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_attn_layers', type=int, default=2)
        parser.add_argument('--num_hops', type=int, default=4)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--warmup_epochs', type=int, default=4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
