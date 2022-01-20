import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from typing import Optional


class SRA(nn.Module):
    """
    Build a SRA based on MoCo (https://arxiv.org/abs/1911.05722) model with: a query encoder, a key encoder,
    and a queue.
    """
    def __init__(
            self,
            base_encoder: Optional[str] = 'resnet18',
            dim: Optional[int] = 128,
            K: Optional[int] = 65536,
            m: Optional[float] = 0.999,
            T: Optional[float] = 0.07,
            n_dataset: Optional[int] = 2,
            device: Optional[float] = "cpu",
            mean_entropy: Optional[bool] = True,
            force_multi_source: Optional[float] = False,
    ):
        """
        base_encoder: func
            Encoder to generate the feature space.
        dim: int
            Dimension of the feature space after projection head. Default value is 128.
        K: int
            Size of the queue which correspond to the number of negative keys. The default value is 65536.
        m: float
            SRA Momentum of updating key encoder. Default value is 0.999.
        T: float
            Softmax temperature. Represent the sharpness / confidence on the prediction. Default value is 0.07.
        device: str
            Device to use. Default value is "cpu". Warning cpu training is really slow.
        mean_entropy: bool
            If True use mean entropy to compute predictions. Otherwise use only the encoder branch.
        force_force_multi_source: bool
            If true, we look for a matching target sample for each source domain. If false, only the best match across
            source domain is considered.
        """
        super(SRA, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.n_dataset = n_dataset
        self.device = device
        self.mean_entropy = mean_entropy
        self.force_multi_source = force_multi_source
        self.loss = nn.CrossEntropyLoss(reduction='sum').to(device=self.device)

        try:
            base_encoder = getattr(models, base_encoder)
        except AttributeError:
            # Attribute not found, use default ResNet18
            base_encoder = models.resnet18

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        self.criterion = nn.CrossEntropyLoss().to(device=self.device)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue for samples and dataset labls
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_dataset", torch.randint(n_dataset, size=(K,)))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_dataset[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, d_set):
        """
        Forward path for input image im_q and im_k. Here we define B = Batch size, H = height of the image, W = width
        of the image, Z = output dimension of the embedding, K = size of the queue.

        Parameters
        ----------
        im_q: Tensor of shape (B, 3, H, W)
            A batch of query images.
        im_k: Tensor of shape (B, 3, H, W)
            A batch of key images.
        d_set: Tensor (B, H, W)
            A batch of dataset assignation (src=0...N-1, tar=N).

        Returns
        -------
        q: Tensor of shape (B, Z)
            Embedding of query images.
        k: Tensor of shape (B, Z)
            Embedding of batch images.
        queue: Tensor of shape (Z, K)
            Queue of previously computed negative examples.
        queue_dataset: Tensor of shape (K, )
            Queue of dataset id of the previously computed negative examples.
        """

        # compute query features and normalize
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():
            # no gradient to keys
            if self.training:
                self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        queue = self.queue.clone().detach()
        queue_dataset = self.queue_dataset.clone().detach()
        # dequeue and enqueue
        if self.training:
            self._dequeue_and_enqueue(k, d_set)

        return q, k, queue, queue_dataset

    def embed(self, im_q):
        """
        Compute embedding of input image im_q

        Parameters
        ----------
        im_q: Tensor of shape (B, 3, H, W)
            A batch of query images.

        Returns
        -------
        q: Tensor of shape (B, Z)
            Embedding of query images.

        """
        # compute query features and normalize
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        return q

    def compute_h_crd(self, q, k, queue):
        """
        Compute cross entropy for each entry of the q embedding wrt the queue.

        Parameters
        ----------
        q: Tensor of shape (B, Z)
            Embedding of the input image.
        queue: Tensor of shape (Z, Q)
            Embedding of the queue.

        Returns
        -------
        h: Tensor of shape (B, )
            Computed entropy based on similarities distributions.

        """
        # Compute entropy between sets predictions
        l_crd = torch.einsum('nc,ck->nk', [q, queue])
        l_crd = torch.exp(l_crd / self.T)
        l_crd = l_crd / l_crd.sum(dim=1).unsqueeze(1)

        l_crd_ = torch.einsum('nc,ck->nk', [k, queue])
        l_crd_ = torch.exp(l_crd_ / self.T)
        l_crd_ = l_crd_ / l_crd_.sum(dim=1).unsqueeze(1)
        # Entropy
        return - torch.sum(l_crd * torch.log(l_crd_ + torch.finfo(float).eps), dim=1)

    def compute_loss_sra(self, q, k, d_set, queue, queue_dataset, s2h_topk_r=1):
        """
        Compute in-domain and cross-domain loss of SRA model. D is the number of datasets. Ni is the number of sample
        from the i-th dataset and Qi the number of negative linked to the i-th dataset.

        Parameters
        ----------
        q: Tensor of shape (B, Z)
            Embedding of query images.
        k: Tensor of shape (B, Z)
            Embedding of batch images.
        d_set: Tensor (B, H, W)
            A batch of dataset assignation (src=0...N-1, tar=N).
        queue: Tensor of shape (Z, K)
            Queue of previously computed negative examples.
        queue_dataset: Tensor of shape (K, )
            Queue of dataset id of the previously computed negative examples.
        s2h_topk_r: float
            Percentage of top k item to consider. Sould be in range [0, 1]. Default value is 0.

        Returns
        -------
        loss_ind: Tensor (1,)
            In-domain loss
        loss_crd: Tensor (1,)
            Cross-domain loss
        logits_ind: List of Tensor of shape (D, Ni, Qi)
            Computed logits (non normalized probabilities).
        labels_ind: List od Tensor of shape (D, Ni)
            Labels for the logit. List of 0s.
        h_crd: List of Tensor of shape (D, Ni)
            Computed entropy for all entries.
        """

        # 1. In-domain Self-supervision
        logits_ind = []
        labels_ind = []
        h_crd = []

        for i in range(self.n_dataset):
            l_ins_pos_ = torch.einsum('nc,nc->n', [q[d_set == i], k[d_set == i]]).unsqueeze(-1)
            l_ins_neg_ = torch.einsum('nc,ck->nk', [q[d_set == i], queue[:, queue_dataset == i]])
            logits_ind_ = torch.cat([l_ins_pos_, l_ins_neg_], dim=1) / self.T
            labels_ind_ = torch.zeros(logits_ind_.shape[0], dtype=torch.long, device=self.device)
            logits_ind.append(logits_ind_)
            labels_ind.append(labels_ind_)

        # 2. Cross-domain self-supervision
        for i in range(self.n_dataset - 1):
            # Cross mapping - last considered as target
            h_crd_di_to_tar = self.compute_h_crd(
                q=q[d_set == i],
                k=k[d_set == i],
                queue=queue[:, queue_dataset == self.n_dataset - 1]
            )

            h_crd_tar_to_di = self.compute_h_crd(
                q=q[d_set == self.n_dataset - 1],
                k=k[d_set == self.n_dataset - 1],
                queue=queue[:, queue_dataset == i]
            )

            # Use mean entropy over both key and query
            if self.mean_entropy:
                h_crd_di_to_tar_ = self.compute_h_crd(
                    q=k[d_set == i],
                    k=q[d_set == i],
                    queue=queue[:, queue_dataset == self.n_dataset - 1]
                )

                h_crd_tar_to_di_ = self.compute_h_crd(
                    q=k[d_set == self.n_dataset - 1],
                    k=q[d_set == self.n_dataset - 1],
                    queue=queue[:, queue_dataset == i]
                )

                h_crd.extend([
                    (1 / 2) * (h_crd_di_to_tar + h_crd_di_to_tar_),
                    (1 / 2) * (h_crd_tar_to_di + h_crd_tar_to_di_)
                ])

            # Use mean entropy using only the query
            if not self.mean_entropy:
                h_crd.extend([
                    h_crd_di_to_tar,
                    h_crd_tar_to_di
                ])

        # 3 Compute size of source and target samples
        n = [l.shape[0] for l in logits_ind]
        ns = np.sum(n[:-1])
        nt = n[-1]

        # 4. In-domain Self-supervision loss
        loss_ind = [self.loss(l, t) for l, t in zip(logits_ind, labels_ind)]
        loss_ind = (1. / (ns + nt)) * torch.stack(loss_ind).sum()

        # 5. Cross-domain self-supervision loss
        if s2h_topk_r != 0:
            # To select the top-k we gather the information across datasets. Meaning if we have 2 source dataset we
            # look for the best target similarities across both source dataset and not for each source.
            if not self.force_multi_source:
                # Option 1 - CRD across domain
                h_crd_s2t = torch.cat(h_crd[0::2])
                h_crd_t2s = torch.cat(h_crd[1::2])
                loss_crc_src_to_tar = h_crd_s2t[h_crd_s2t.argsort()[:int(s2h_topk_r * len(h_crd_s2t))]].sum(dim=0)
                loss_crc_tar_to_src = h_crd_t2s[h_crd_t2s.argsort()[:int(s2h_topk_r * len(h_crd_t2s))]].sum(dim=0)
                loss_crd = torch.stack([loss_crc_src_to_tar, loss_crc_tar_to_src]).sum()
            else:
                # Option 2 - CRD for each entry
                loss_crd = [h[h.argsort()[:int(s2h_topk_r * len(h))]].sum(dim=0) for h in h_crd]
                loss_crd = torch.stack(loss_crd).sum()
            # Aggregate results
            loss_crd = (1. / (ns + (self.n_dataset-1) * nt) / s2h_topk_r) * loss_crd.sum()
        else:
            loss_crd = torch.tensor(0, dtype=torch.float32, device=self.device)

        return loss_ind, loss_crd, logits_ind, labels_ind, h_crd

    def compute_loss_moco(self, q, k, queue):
        """
         Compute the loss of the MoCo model. D is the number of datasets. Ni is the number of sample
         from the i-th dataset and Qi the number of negative linked to the i-th dataset.

         Parameters
         ----------
         q: Tensor of shape (B, Z)
             Embedding of query images.
         k: Tensor of shape (B, Z)
             Embedding of batch images.
         queue: Tensor of shape (Z, K)
             Queue of previously computed negative examples.

         Returns
         -------
         loss: Tensor (1,)
             MoCo loss
        logits_ind: List of Tensor of shape (D, Ni, Qi)
            Computed logits (non normalized probabilities).
        labels_ind: List od Tensor of shape (D, Ni)
            Labels for the logit. List of 0s.
         """

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        return self.criterion(logits, labels), [logits], [labels]


class SRACls(nn.Module):
    """
    Build a SRA based on MoCo (https://arxiv.org/abs/1911.05722) model with: a query encoder, a key encoder,
    and a queue.
    """
    def __init__(self, base_encoder='resnet18', n_cls=2, dim=128, device="cpu"):
        """
        base_encoder: str
            Encoder to generate the feature space.
        n_cls: int
            Number of output classes. Default value is 2.
        dim: int
            Dimension of the feature space after projection head. Default value is 128.
        device: str
            Device to use. Default value is "cpu". Warning cpu training is really slow.
        """
        super(SRACls, self).__init__()

        self.device = device

        # create the encoders
        # num_classes is the output fc dimension
        try:
            base_encoder = getattr(models, base_encoder)
        except AttributeError:
            # Attribute not found, use default ResNet18
            base_encoder = models.resnet18
        self.encoder_q = base_encoder(num_classes=dim)

        dim_encoder = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_encoder, dim_encoder),
            nn.ReLU(),
            self.encoder_q.fc,
            nn.Linear(dim, n_cls),
        )

    def forward(self, im):
        """
        Forward path for input image im_q and im_k. Here we define B = Batch size, H = height of the image, W = width
        of the image, N_cls = number of output classes.

        Parameters
        ----------
        im: Tensor of shape (B, 3, H, W)
            A batch of images.

        Returns
        -------
        y: Tensor of shape (B, n_cls)
            Embedding of query images.
        """

        # compute query features and normalize
        y = self.encoder_q(im)  # queries: NxC
        return y


class ResNetCls(nn.Module):
    """
    Build a SRA based on MoCo (https://arxiv.org/abs/1911.05722) model with: a query encoder, a key encoder,
    and a queue.
    """
    def __init__(self, base_encoder='resnet18', n_cls=2, device="cpu"):
        """
        base_encoder: str
            Encoder to generate the feature space.
        n_cls: int
            Number of output classes. Default value is 2.
        dim: int
            Dimension of the feature space after projection head. Default value is 128.
        device: str
            Device to use. Default value is "cpu". Warning cpu training is really slow.
        """

        super(ResNetCls, self).__init__()
        self.device = device

        # create the encoders
        # num_classes is the output fc dimension
        try:
            base_encoder = getattr(models, base_encoder)
        except AttributeError:
            # Attribute not found, use default ResNet18
            base_encoder = models.resnet18
        self.encoder_q = base_encoder(pretrained=True)

        dim_encoder = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_encoder, n_cls),
        )

    def forward(self, im):
        """
        Forward path for input image im_q and im_k. Here we define B = Batch size, H = height of the image, W = width
        of the image, N_cls = number of output classes.

        Parameters
        ----------
        im: Tensor of shape (B, 3, H, W)
            A batch of images.

        Returns
        -------
        y: Tensor of shape (B, n_cls)
            Embedding of query images.
        """

        # compute query features and normalize
        y = self.encoder_q(im)  # queries: NxC
        return y