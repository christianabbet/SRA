import torch
import torch.nn as nn


class SRA(nn.Module):
    """
    Build a SRA based on MoCo (https://arxiv.org/abs/1911.05722) model with: a query encoder, a key encoder,
    and a queue.
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, use_hema=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        use_hema: .... (default: False)
        """
        super(SRA, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.use_hema = use_hema

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        self.dim_encoder = self.encoder_q.fc.weight.shape[1]

        # reset avg pooling and fc
        self.encoder_q.avgpool = nn.Sequential()
        self.encoder_q.fc = nn.Sequential()
        self.encoder_k.avgpool = nn.Sequential()
        self.encoder_k.fc = nn.Sequential()

        # create projection head (mlp classifier)
        self.encoder_q_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder), nn.BatchNorm1d(self.dim_encoder), nn.ReLU(),
            nn.Linear(self.dim_encoder, dim))
        self.encoder_k_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder), nn.BatchNorm1d(self.dim_encoder), nn.ReLU(),
            nn.Linear(self.dim_encoder, dim))

        # Initialize encoder k with q weights
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.encoder_q_mlp.parameters(), self.encoder_k_mlp.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if self.use_hema:
            self.decoder_hema = SimpleDecoder(hidden_dimension=self.dim_encoder, chanel_out=1)

        # create the queue for samples and dataset labls
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_dataset", torch.randint(2, size=(K,)))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.encoder_q_mlp.parameters(), self.encoder_k_mlp.parameters()):
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
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            d_set: a batch of dataset assignation (src=0, tar=1)
        Output:
            logits_ind_d0:  logits for in-domain loss over source (d0) samples
            labels_ind_d0:  labels for in-domain loss over source (d0) samples
            logits_ind_d1:  logits for in-domain loss over target (d1) samples
            labels_ind_d1:  label for in-domain loss over target (d1) samples
            h_crd_d0tod1:   entropy measure for cross-domain loss from source (d0) to target (d1)
            h_crd_d1tod0:   entropy measure for cross-domain loss from target (d1) to source (d0)
        """

        # compute query features and normalize
        qh = self.encoder_q(im_q)  # queries: NxC
        qh = qh.view((-1, self.dim_encoder, 7, 7))
        q = nn.functional.adaptive_avg_pool2d(qh, (1, 1)).squeeze()
        q = self.encoder_q_mlp(q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            kh = self.encoder_k(im_k)  # keys: NxC
            kh = kh.view((-1, self.dim_encoder, 7, 7))
            k = nn.functional.adaptive_avg_pool2d(kh, (1, 1)).squeeze()
            k = self.encoder_q_mlp(k)
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        queue = self.queue.clone().detach()
        queue_dataset = self.queue_dataset.clone().detach()

        # 1. In-domain Self-supervision
        l_ins_pos_d0 = torch.einsum('nc,nc->n', [q[d_set == 0], k[d_set == 0]]).unsqueeze(-1)
        l_ins_pos_d1 = torch.einsum('nc,nc->n', [q[d_set == 1], k[d_set == 1]]).unsqueeze(-1)

        l_ins_neg_d0 = torch.einsum('nc,ck->nk', [q[d_set == 0], queue[:, queue_dataset == 0]])
        l_ins_neg_d1 = torch.einsum('nc,ck->nk', [q[d_set == 1], queue[:, queue_dataset == 1]])

        # logits: Nx(1+K)
        logits_ind_d0 = torch.cat([l_ins_pos_d0, l_ins_neg_d0], dim=1) / self.T
        logits_ind_d1 = torch.cat([l_ins_pos_d1, l_ins_neg_d1], dim=1) / self.T

        # labels: positive key indicators
        labels_ind_d0 = torch.zeros(logits_ind_d0.shape[0], dtype=torch.long).cuda()
        labels_ind_d1 = torch.zeros(logits_ind_d1.shape[0], dtype=torch.long).cuda()

        # 2. Cross-domain self-supervision
        h_crd_d0tod1 = self.compute_h_crd(q=q[d_set == 0],
                                          queue=queue[:, queue_dataset == 1])
        h_crd_d1tod0 = self.compute_h_crd(q=q[d_set == 1],
                                          queue=queue[:, queue_dataset == 0])

        # 3. Reconstruction loss is defined
        if self.use_hema:
            logits_hema = self.decoder_hema(qh)
        else:
            logits_hema = None

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, d_set)

        return logits_ind_d0, labels_ind_d0, logits_ind_d1, labels_ind_d1, h_crd_d0tod1, h_crd_d1tod0, logits_hema

    def compute_h_crd(self, q, queue):

        # Compute entropy between sets predictions
        l_crd = torch.einsum('nc,ck->nk', [q, queue])
        l_crd = torch.exp(l_crd / self.T)
        l_crd = l_crd / l_crd.sum(dim=1).unsqueeze(1)
        # Entropy
        return - torch.sum(l_crd * torch.log(l_crd + torch.finfo(float).eps), dim=1)


class SimpleDecoder(nn.Module):

    def __init__(self, hidden_dimension=512, chanel_out=3):
        super(SimpleDecoder, self).__init__()

        self.conv_up_5 = nn.Conv2d(hidden_dimension, hidden_dimension//2, 3, padding=1)
        self.conv_up_4 = nn.Conv2d(hidden_dimension//2, hidden_dimension//4, 3, padding=1)
        self.conv_up_3 = nn.Conv2d(hidden_dimension//4, hidden_dimension//8, 3, padding=1)
        self.conv_up_2 = nn.Conv2d(hidden_dimension//8, hidden_dimension//16, 3, padding=1)
        self.conv_up_1 = nn.Conv2d(hidden_dimension//16, hidden_dimension//32, 5, padding=2)
        self.decoder = nn.Conv2d(hidden_dimension//32, chanel_out, 5, padding=2)

    def forward(self, z):

        h = nn.ReLU()(self.conv_up_5(z))
        h = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)(h)
        h = nn.ReLU()(self.conv_up_4(h))
        h = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)(h)
        h = nn.ReLU()(self.conv_up_3(h))
        h = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)(h)
        h = nn.ReLU()(self.conv_up_2(h))
        h = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)(h)
        h = nn.ReLU()(self.conv_up_1(h))
        h = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)(h)
        x_hat = nn.Sigmoid()(self.decoder(h))

        return x_hat