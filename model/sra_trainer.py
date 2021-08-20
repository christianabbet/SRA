from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from model.utils import get_logger
import os
from numpy import ndarray
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.utils import accuracy_topk


class SRATrainer:

    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            use_lind: Optional[bool] = True,
            use_lcrd: Optional[bool] = True,
            use_e2h: Optional[bool] = True,
            use_moco: Optional[bool] = False,
            opt_lr: Optional[float] = 0.03,
            opt_momentum: Optional[float] = 0.9,
            opt_weight_decay: Optional[float] = 1e-4,
            t_max: Optional[int] = 200,
            sh: Optional[float] = 0.2,
            sw: Optional[float] = 0.25,
            checkpoint_epochs: Optional[int] = 50,
            device: Optional[str] = "cpu",
            logger: Optional[object] = None,
            prefix: Optional[str] = None,
            loadpath: Optional[str] = None,
    ) -> None:
        """
        Create trainer for histopathology dataset.

        Parameters
        ----------
        model: callable
            Model to train.
        train_loader: callable
            Dataloader for training data.
        val_loader: callable
            Dataloader for validation data.
        use_lind: bool, optional
            Whether to use or not in-domain learning
        use_lcrd: bool, optional
            Whether to use or not cross-domain learning
        use_e2h: bool, optional
            Whether to use or not easy-to-hard learning
        use_moco: bool, optional
            Compute the loss w.r.t. the queue as in MoCo. If True then ind, crd and e2h are ignored.
        opt_lr: float
            SGD learning rate. Default value is 0.03.
        opt_momentum: float
            SGD momentum. Default value is 0.9,
        opt_weight_decay: float
            SDG weights decay. Default value is 1e-4.
        t_max: int
            T max for the cosine LR scheduler. Should be equal to the number of epochs. Default value is 200.
        sh: float
            Step function height of the easy-to-hard. Should be in range ]0, 1]. See r formula. Default value is 0.2.
        sw: float
            Step function width of the easy-to-hard. Should be in range ]0, 1]. See r formula. Default value is 0.25.
        checkpoint_epochs: int
            Number of epochs before checkpoint.
        device: str, optional
            Choose whether to use cuda for training or not.
        logger: object, optional
            Logger to use for output. If None, create a new one.
        prefix: str, optional
            Prefix to add in front of file to discriminate them. Default value is empty
        loadpath: str, optional
            Path to pretrained model.

        Notes
        -----
        ..math: r = np.floor(epoch / (sw * n_epochs)) * sh
        """

        # Create logger is needed
        self.logger = logger
        if self.logger is None:
            self.logger = get_logger()

        # Model
        self.prefix = prefix
        self.device = device
        self.use_lind = use_lind
        self.use_lcrd = use_lcrd
        self.use_e2h = use_e2h
        self.use_moco = use_moco
        self.sh = sh
        self.sw = sw
        self.checkpoint_epochs = checkpoint_epochs
        self.model = model.to(device=self.device)

        # Data loader
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer and objective
        self.optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=opt_lr,
            momentum=opt_momentum,
            weight_decay=opt_weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=0)

        # Load path if existing
        self.load(loadpath)

        # Output files
        self.writer = SummaryWriter(comment=prefix)

    def load(self, loadpath: str) -> None:

        if loadpath is None:
            self.logger.error('No weights provided ...')
            return

        # Load path if existing
        if os.path.exists(loadpath):
            self.logger.debug('Loading weights from: {} ...'.format(loadpath))
            stat = torch.load(loadpath)
            self.model.load_state_dict(stat['model_state_dict'])
            self.optimizer.load_state_dict(stat['optimizer_state_dict'])
            # For old models
            # self.model.load_state_dict(stat['state_dict'])
        else:
            self.logger.error('Weights not found: {}'.format(loadpath))
            raise Exception("Non-valid loading path ...")

    @staticmethod
    def init_seed(seed: Optional[int] = 0) -> None:
        """
        Set seed for torch, numpy and random for reproducibility.

        Parameters
        ----------
        seed: int, optional
            Seed value for reproducibility. Default value is 0.

        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self, n_epochs: Optional[int] = 200) -> None:
        """
        Train model for a certain numbr of epochs.

        Parameters
        ----------
        n_epochs: int, optional
            Number of epochs to train. Default value is 200.
        """

        # Initialize seeds and loss max
        self.init_seed()

        self.logger.debug('use_lind: {}'.format(self.use_lind))
        self.logger.debug('use_lcrd: {}'.format(self.use_lcrd))
        self.logger.debug('use_e2h: {}'.format(self.use_e2h))
        self.logger.debug('use_moco: {}'.format(self.use_moco))

        # Iterate over epochs
        for epoch in range(n_epochs):

            losses = []
            losses_ind = []
            losses_crd = []
            accs_top1 = []
            accs_top5 = []

            if self.use_e2h:
                s2h_topk_r = np.floor(epoch / (self.sw * n_epochs)) * self.sh
            else:
                s2h_topk_r = 1.0
            self.logger.debug('Simple-to-hard consider top: {:.1f}%'.format(s2h_topk_r * 100))

            self.model.train()

            for images, d_set in tqdm(self.train_loader, desc="Train ..."):
                # images[0]: key, image[1]: query, d_set: label source (0) or target (1)
                images[0] = images[0].to(device=self.device)
                images[1] = images[1].to(device=self.device)
                d_set = d_set.to(device=self.device)

                q, k, queue, queue_dataset = self.model(
                    im_q=images[0],
                    im_k=images[1],
                    d_set=d_set,
                )

                try:
                    # Hot fix
                    if not self.use_moco:
                        loss_ind, loss_crd, l_ind, t_ind, _ = self.model.compute_loss_sra(
                            q=q,
                            k=k,
                            d_set=d_set,
                            queue=queue,
                            queue_dataset=queue_dataset,
                            s2h_topk_r=s2h_topk_r
                        )
                    else:
                        loss_ind, l_ind, t_ind = self.model.compute_loss_moco(
                            q=q,
                            k=k,
                            queue=queue
                        )
                        loss_crd = torch.tensor(0, dtype=torch.float32, device=self.device)

                except Exception as e:
                    continue

                # 3. Overall loss
                loss = torch.tensor(0, dtype=torch.float32, device=self.device)
                if self.use_lind:
                    loss += loss_ind
                if self.use_lcrd:
                    loss += loss_crd

                # 5. Compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses_ind.append(loss_ind.item())
                losses_crd.append(loss_crd.item())
                losses.append(loss.item())
                acc_top1 = [accuracy_topk(l, t, topk=(1,))[0].item() for (l, t) in zip(l_ind, t_ind)]
                acc_top5 = [accuracy_topk(l, t, topk=(5,))[0].item() for (l, t) in zip(l_ind, t_ind)]
                accs_top1.append(acc_top1)
                accs_top5.append(acc_top5)

            # Scheduler step
            self.scheduler.step()

            losses = np.mean(losses)
            losses_ind = np.mean(losses_ind)
            losses_crd = np.mean(losses_crd)
            accs_top1 = np.mean(accs_top1, axis=0)
            accs_top5 = np.mean(accs_top5, axis=0)

            # Log writer
            loss_eval, loss_ind_eval, loss_crd_eval, accs_top1_eval, accs_top5_eval = self.eval()
            self.logger.debug('Epoch {}'.format(epoch))
            self.logger.debug('Loss train: {:.3f}, {:.3f}, {:.3f}'.format(losses, losses_ind, losses_crd))
            self.logger.debug('Loss eval: {:.3f}, {:.3f}, {:.3f}'.format(loss_eval, loss_ind_eval, loss_crd_eval))
            self.logger.debug('Acc top1 in-domain train: {}'.format(", ".join(["D{}: {:.1f}%".format(i, accs_top1[i]) for i in range(accs_top1.shape[0])])))
            self.logger.debug('Acc top5 in-domain train: {}'.format(", ".join(["D{}: {:.1f}%".format(i, accs_top5[i]) for i in range(accs_top5.shape[0])])))
            self.logger.debug('Acc top1 in-domain eval: {}'.format(", ".join(["D{}: {:.1f}%".format(i, accs_top1_eval[i]) for i in range(accs_top1_eval.shape[0])])))
            self.logger.debug('Acc top5 in-domain eval: {}'.format(", ".join(["D{}: {:.1f}%".format(i, accs_top5_eval[i]) for i in range(accs_top5_eval.shape[0])])))

            self.writer.add_scalars('Loss', {'train': losses, 'val': loss_eval}, epoch)
            self.writer.add_scalars('LossIND', {'train': losses_ind, 'val': loss_ind_eval}, epoch)
            self.writer.add_scalars('LossCRD', {'train': losses_crd, 'val': loss_crd_eval}, epoch)
            self.writer.add_scalars('AccTop1', {"train_D{}".format(i): accs_top1[i] for i in range(len(accs_top1))}, epoch)
            self.writer.add_scalars('AccTop5', {"train_D{}".format(i): accs_top5[i] for i in range(len(accs_top5))}, epoch)
            self.writer.add_scalars('AccTop1', {"val_D{}".format(i): accs_top1_eval[i] for i in range(len(accs_top1_eval))}, epoch)
            self.writer.add_scalars('AccTop5', {"val_D{}".format(i): accs_top5_eval[i] for i in range(len(accs_top5_eval))}, epoch)

            # Check is loss improved to save model
            # if loss_eval < best_valid_losses:
            #     best_valid_losses = loss_eval
            #     self.save(path="best_model_{}_{}.pth".format(epoch, self.prefix))

            # Save model each certain amount of epochs
            if (epoch + 1) % self.checkpoint_epochs == 0:
                self.save(path="checkpoint_{}_{}.pth".format(epoch, self.prefix))

    def eval(self):
        """
        Evaluate model on dataset

        Returns
        -------
        y_preds: ndarray
            Output prediction probabilities.
        y_labels: ndarray
            Classes ground truth.
        loss: float
            Average loss over all batches.
        accuracy: float
            Average accuracy over all batches.
        embed: bool, optional
            If True return embedding dimension, if False, return classification decision.
        """

        self.model.eval()
        with torch.no_grad():

            losses = []
            losses_ind = []
            losses_crd = []
            accs_top1 = []
            accs_top5 = []

            for images, d_set in tqdm(self.val_loader, desc='Validation ...'):
                # images[0]: key, image[1]: query, d_set: label source (0) or target (1)
                images[0] = images[0].to(device=self.device)
                images[1] = images[1].to(device=self.device)
                d_set = d_set.to(device=self.device)

                q, k, queue, queue_dataset = self.model(
                    im_q=images[0],
                    im_k=images[1],
                    d_set=d_set,
                )

                try:
                    loss_ind, loss_crd, l_ind, t_ind, _ = self.model.compute_loss_sra(
                        q=q,
                        k=k,
                        d_set=d_set,
                        queue=queue,
                        queue_dataset=queue_dataset,
                        s2h_topk_r=1.0
                    )
                except Exception:
                    continue

                # 3. Overall loss
                loss = torch.tensor(0, dtype=torch.float32, device=self.device)
                if self.use_lind:
                    loss += loss_ind
                if self.use_lcrd:
                    loss += loss_crd

                losses_ind.append(loss_ind.item())
                losses_crd.append(loss_crd.item())
                losses.append(loss.item())
                acc_top1 = [accuracy_topk(l, t, topk=(1,))[0].item() for (l, t) in zip(l_ind, t_ind)]
                acc_top5 = [accuracy_topk(l, t, topk=(5,))[0].item() for (l, t) in zip(l_ind, t_ind)]
                accs_top1.append(acc_top1)
                accs_top5.append(acc_top5)

            losses = np.mean(losses)
            losses_ind = np.mean(losses_ind)
            losses_crd = np.mean(losses_crd)
            accs_top5 = np.mean(accs_top5, axis=0)
            accs_top1 = np.mean(accs_top1, axis=0)

        return losses, losses_ind, losses_crd, accs_top1, accs_top5

    def embed(self, loader: DataLoader):
        """
        Embedding model on dataset

        Parameters
        ----------
        loader: DataLoader
            Dataset loader to use to compute embedding.

        Returns
        -------
        embedding: ndarray
            Embedding of the data.
        labels: ndarray
            Classes ground truth.
        """

        self.model.eval()
        with torch.no_grad():

            embedding = []
            labels = []

            for images, label in tqdm(loader, desc='Embedding ...'):
                # images[0]: key, image[1]: query, d_set: label source (0) or target (1)
                images = images.to(device=self.device)

                z = self.model.embed(
                    im_q=images,
                )

                embedding.extend(z.detach().cpu().numpy())
                labels.extend(label)

        return embedding, labels

    def save(self, path) -> None:
        """
        Save model and optimizer state.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
