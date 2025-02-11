import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoints.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TrainerBase(nn.Module):
    def __init__(self,
                 epoches,
                 train_loader,
                 optimizer,
                 device,
                 IFEarlyStopping,
                 IFadjust_learning_rate,
                 **kwargs):
        super(TrainerBase, self).__init__()

        self.epoches = epoches
        if self.epoches is None:
            raise ValueError("Please input the total number of training iterations")

        self.train_loader = train_loader
        if self.train_loader is None:
            raise ValueError("Please input train_loader")

        self.optimizer = optimizer
        if self.optimizer is None:
            raise ValueError("Please pass in the optimizer")
        self.device = device
        if self.device is None:
            raise ValueError("Please input the type of device")

        # If early stopping is enabled, a series of following judgments must be made
        self.IFEarlyStopping = IFEarlyStopping
        if IFEarlyStopping:
            if "patience" in kwargs.keys():
                self.early_stopping = EarlyStopping(patience=kwargs["patience"], verbose=True)
            else:
                raise ValueError("Enabling early stopping strategy requires input {patience=int X}")

            if "val_loader" in kwargs.keys():
                self.val_loader = kwargs["val_loader"]
            else:
                raise ValueError("To enable the early stopping strategy, a validation set must be provided val_loader")

        # If a learning rate adjustment strategy is enabled, a series of following judgments must be made
        self.IFadjust_learning_rate = IFadjust_learning_rate
        if IFadjust_learning_rate:
            if "types" in kwargs.keys():
                self.types = kwargs["types"]
                if "lr_adjust" in kwargs.keys():
                    self.lr_adjust = kwargs["lr_adjust"]
                else:
                    self.lr_adjust = None
            else:
                raise ValueError("To enable a learning rate adjustment strategy, one must start from {type1 or type2}")

    def adjust_learning_rate(self, epoch, learning_rate):
        # lr = args.learning_rate * (0.2 ** (epoch // 2))
        if self.types == 'type1':
            lr_adjust = {epoch: learning_rate * (0.1 ** ((epoch - 1) // 10))}
        elif self.types == 'type2':
            if self.lr_adjust is not None:
                lr_adjust = self.lr_adjust
            else:
                lr_adjust = {
                    5: 1e-4, 10: 5e-5, 20: 1e-5, 25: 5e-6,
                    30: 1e-6, 35: 5e-7, 40: 1e-8
                }
        else:
            raise ValueError("Please select from {{0}or{1}}".format("type1", "type2"))

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    @staticmethod
    def save_best_model(model, path):
        torch.save(model.state_dict(), path + '/' + 'BestModel.pth')
        print("Successfully saved the trained model (in .pth format) to:" + str(path))

    def forward(self, model, *args, **kwargs):

        pass


class SimpleDiffusionTrainer(TrainerBase):
    def __init__(self,
                 epoches=None,
                 train_loader=None,
                 optimizer=None,
                 device=None,
                 IFEarlyStopping=False,
                 IFadjust_learning_rate=False,
                 **kwargs):
        super(SimpleDiffusionTrainer, self).__init__(epoches, train_loader, optimizer, device,
                                                     IFEarlyStopping, IFadjust_learning_rate,
                                                     **kwargs)

        if "timesteps" in kwargs.keys():
            self.timesteps = kwargs["timesteps"]
        else:
            raise ValueError("Diffusion model training must provide the diffusion steps parameter")

    def forward(self, model, *args, **kwargs):
        loss_min = float('inf')
        for i in range(self.epoches):
            losses = []
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for step, (_, gt) in loop:
                features = gt.to(self.device)
                batch_size = features.shape[0]

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                # loss = model(mode="train", x_start=features, t=t, loss_type="huber")
                loss = model(mode="train", x_start=features, t=t, loss_type="huber")
                losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update
                loop.set_description(f'Epoch [{i}/{self.epoches}]')
                loop.set_postfix(loss=loss.item())
            loss_total = sum(losses) / len(losses)
            if "model_save_path" in kwargs.keys() and loss_total < loss_min:
                self.save_best_model(model=model, path=kwargs["model_save_path"])
                loss_min = loss_total
        # if "model_save_path" in kwargs.keys():
        #     self.save_best_model(model=model, path=kwargs["model_save_path"])

        return model


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(588, 10)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out
