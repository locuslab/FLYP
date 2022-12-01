import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(image_features,
                    text_features,
                    local_loss=False,
                    gather_with_grad=False,
                    rank=0,
                    world_size=1,
                    use_horovod=False):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self,
                image_features,
                text_features,
                logit_scale,
                ground_labels=None,
                ignore=False,
                google_sup_loss=False):
        assert not (ignore and google_sup_loss), 'please specify only one'
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features, self.local_loss,
                self.gather_with_grad, self.rank, self.world_size,
                self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            # import pdb;pdb.set_trace()
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]

        if ground_labels is not None:
            ground_labels_repeated = ground_labels.view(1, -1).repeat(
                image_features.shape[0], 1)
            equal_labels = (ground_labels_repeated == ground_labels.view(
                -1, 1)).type(torch.float)
            # equal_labels = torch.eye(equal_labels.shape[0],
            #                          device=device,
            #                          dtype=torch.float)

            if ignore:
                I = torch.eye(equal_labels.shape[0],
                              device=device,
                              dtype=torch.float)
                labels = I - 100 * (equal_labels - I)

                image_logit_exp = torch.exp(
                    logits_per_image -
                    torch.max(logits_per_image, dim=1, keepdim=True).values)
                text_logit_exp = torch.exp(
                    logits_per_text -
                    torch.max(logits_per_text, dim=1, keepdim=True).values)

                image_logit_exp = image_logit_exp * (labels != -100)
                text_logit_exp = text_logit_exp * (labels != -100)

                image_logit_exp = torch.diagonal(image_logit_exp) / torch.sum(
                    image_logit_exp, dim=1)
                text_logit_exp = torch.diagonal(text_logit_exp) / torch.sum(
                    text_logit_exp, dim=1)

                image_logit_exp = -torch.log(image_logit_exp)
                text_logit_exp = -torch.log(text_logit_exp)

                total_loss = torch.mean(image_logit_exp) + torch.mean(
                    text_logit_exp)

                total_loss /= 2
            elif google_sup_loss:
                image_logit_exp = torch.exp(
                    logits_per_image -
                    torch.max(logits_per_image, dim=1, keepdim=True).values)
                image_sum = torch.sum(image_logit_exp, dim=1, keepdim=True)
                image_sum = image_sum.repeat(1, image_logit_exp.shape[1])
                image_sum_sub = image_sum - image_logit_exp
                image_logit_exp /= image_sum_sub
                image_logit_exp = -torch.log(image_logit_exp)
                image_logit_exp *= equal_labels
                loss1 = torch.sum(image_logit_exp, dim=1) / torch.sum(
                    equal_labels, dim=1)
                loss1 = torch.mean(loss1)

                text_logit_exp = torch.exp(
                    logits_per_text -
                    torch.max(logits_per_text, dim=1, keepdim=True).values)
                text_sum = torch.sum(text_logit_exp, dim=1, keepdim=True)
                text_sum = text_sum.repeat(1, text_logit_exp.shape[1])
                text_sum_sub = text_sum - text_logit_exp
                text_logit_exp /= text_sum_sub
                text_logit_exp = -torch.log(text_logit_exp)
                text_logit_exp *= equal_labels
                loss2 = torch.sum(text_logit_exp, dim=1) / torch.sum(
                    equal_labels, dim=1)
                loss2 = torch.mean(loss2)

                total_loss = (loss1 + loss2) / 2
            else:
                labels = equal_labels / torch.sum(equal_labels, dim=1).view(
                    -1, 1)
                total_loss = (F.cross_entropy(logits_per_image, labels) +
                              F.cross_entropy(logits_per_text, labels)) / 2

        else:
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits,
                                      device=device,
                                      dtype=torch.long)

                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]

            total_loss = (F.cross_entropy(logits_per_image, labels) +
                          F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss
