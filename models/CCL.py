import torch
from torch import nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, num_capsules, capsule_dim):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.W = nn.Parameter(torch.randn(num_capsules, input_dim, capsule_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1).expand(-1, self.num_capsules, -1).to(self.W.device)
        W = self.W.unsqueeze(0)
        x = x.float()
        W = W.float()
        u_hat = torch.matmul(x, W)  # (batch_size, input_dim * num_capsules, capsule_dim)
        u_hat = u_hat.squeeze(2)
        b = torch.zeros_like(u_hat[:, :, 0])  # (batch_size, input_dim * num_capsules)
        c = F.softmax(b, dim=1)  # routing coefficients
        # Expand c to match the shape of u_hat
        s = torch.sum(c.unsqueeze(-1) * u_hat, dim=1)  # (batch_size, num_capsules, capsule_dim)
        v = self.squash(s)  # (batch_size, num_capsules, capsule_dim)
        return v

    def squash(self, s):
        magnitude = torch.sum(s ** 2, dim=-1, keepdim=True)
        scale = magnitude / (1 + magnitude)
        return scale * s / torch.sqrt(magnitude)

class Conloss(nn.Module):
    def __init__(self, opt, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(Conloss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.capsule_layer = CapsuleLayer(input_dim=1, num_capsules=1, capsule_dim=768)

    def forward(self, opt, features, labels=None, mask=None, supcon_s=False, selfcon_s_FG=False, selfcon_m_FG=False):

        features = features.unsqueeze(dim=1)
        features = features.unsqueeze(dim=2)
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0] if not selfcon_m_FG else int(features.shape[0] / 2)

        cap_out = self.capsule_layer(features)
        cap_out = cap_out.to(features.device)
        features = cap_out.float()
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)
        else:
            mask = mask.float().to(device)
        if not selfcon_s_FG and not selfcon_m_FG:
            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            if self.contrast_mode == 'one':
                anchor_feature = features[:, 0]
                anchor_count = 1
            elif self.contrast_mode == 'all':
                anchor_feature = contrast_feature
                anchor_count = contrast_count
            else:
                raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        elif selfcon_s_FG:
            contrast_count = features.shape[1]
            anchor_count = features.shape[1] - 1
            anchor_feature, contrast_feature = torch.cat(torch.unbind(features, dim=1)[:-1], dim=0), \
                torch.unbind(features, dim=1)[-1]
            contrast_feature = torch.cat([anchor_feature, contrast_feature], dim=0)
        elif selfcon_m_FG:
            contrast_count = int(features.shape[1] * 2)
            anchor_count = (features.shape[1] - 1) * 2
            anchor_feature, contrast_feature = torch.cat(torch.unbind(features, dim=1)[:-1], dim=0), \
                torch.unbind(features, dim=1)[-1]
            contrast_feature = torch.cat([anchor_feature, contrast_feature], dim=0)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature.float(), contrast_feature.transpose(0, 1).float()),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        if supcon_s:
            idx = mask.sum(1) != 0
            mask = mask[idx, :]
            logits_mask = logits_mask[idx, :]
            logits = logits[idx, :]
            batch_size = idx.sum()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss




