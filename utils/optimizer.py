
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

def get_optimizer(hparams, models):
    eps = 1e-5
    parameters = []
    for model in models:
        parameters += list(model.parameters())
    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lrate, 
                        weight_decay=1e-5)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(parameters, lr=hparams.lrate,
                        betas=(0.9, 0.999),
                        weight_decay=1e-5)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer

def get_scheduler(hparams, optimizer):
    eps = 1e-5
    # if hparams.lr_scheduler == 'steplr':
    #     scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, 
    #                             gamma=hparams.decay_gamma)
    # elif hparams.lr_scheduler == 'cosine':
    #     scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_steps, eta_min=eps)

    # else:
    #     raise ValueError('scheduler not recognized!')
    num_steps = hparams.num_steps
    if hparams.ddp:
        num_steps /= 8
    return CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=eps)