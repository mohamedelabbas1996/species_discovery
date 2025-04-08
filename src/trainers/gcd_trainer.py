from .base_trainer import BaseTrainer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import Config
from torch.optim import SGD, lr_scheduler
from timm.utils import AverageMeter

device = "cuda" if torch.cuda.is_available() else "cpu"


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def info_nce_logits(features, config):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(config.n_views)], dim=0)

    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / config.temperature
    return logits, labels


class GCDTrainer(BaseTrainer):
    def __init__(
        self,
        net: nn.Module,
        train_loader: DataLoader,
        config: Config,
    ) -> None:

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        if config.network.name == "dinov2":
            for m in net.parameters():
                m.requires_grad = False

        # Only finetune layers from block 'config.grad_from_block' onwards
        for name, m in net.named_parameters():

            if "block" in name:
                block_num = int(name.split(".")[2])
                if block_num >= config.grad_from_block:
                    m.requires_grad = True

        self.net = net
        self.train_loader = train_loader
        self.projection_head = None
        self.config = config

        self.optimizer = SGD(
            self.net.parameters(),
            lr=self.config.optimizer.lr,
            momentum=self.config.optimizer.momentum,
            weight_decay=self.config.optimizer.weight_decay,
        )

        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.optimizer.lr * 1e-3,
        )

        self.sup_con_crit = SupConLoss()

    def train_epoch(self, epoch_idx):
        train_loader = self.train_loader
        self.net.train()

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        for batch in tqdm(
            train_loader,
            desc="Epoch {:03d}: ".format(epoch_idx),
            position=0,
            leave=True,
        ):
            # images, class_labels, uq_idxs, mask_lab = batch
            # mask_lab = mask_lab[:, 0]

            # class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()

            data = batch["data"]
            class_labels = batch["label"].to(device)
            images = torch.cat(data, dim=0).to(device)

            mask_lab = class_labels != -1

            _, features = self.net(images, return_both_features=True)

            # L2-normalize features
            features = torch.nn.functional.normalize(features, dim=-1)

            # Choose which instances to run the contrastive loss on
            if self.config.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                # Contrastive loss for all examples
                con_feats = features

            contrastive_logits, contrastive_labels = info_nce_logits(
                features=con_feats, config=self.config
            )
            contrastive_loss = torch.nn.CrossEntropyLoss()(
                contrastive_logits, contrastive_labels
            )

            # Supervised contrastive loss
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]

            sup_con_loss = self.sup_con_crit(sup_con_feats, labels=sup_con_labels)

            # Total loss
            loss = (
                1 - self.config.sup_con_weight
            ) * contrastive_loss + self.config.sup_con_weight * sup_con_loss

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            loss_record.update(loss.item(), class_labels.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(
            "Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} ".format(
                epoch_idx, loss_record.avg, train_acc_record.avg
            )
        )

        metrics = {}
        metrics["epoch_idx"] = epoch_idx
        metrics["loss"] = loss_record.avg
        metrics["acc"] = train_acc_record.avg

        return self.net, metrics
