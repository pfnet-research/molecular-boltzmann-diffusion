import torch
from torch.optim.adam import Adam
from loguru import logger

from moldiv.util import (
    get_beta_i,
    set_device_around,
)
from moldiv.twodim import (
    sample_data,
    get_dataloader,
    get_extension_manager,
    ScoreModel,
    train,
)

use_cuda = torch.cuda.is_available()
device = set_device_around(use_cuda, seed=0)
model = ScoreModel().to(device)
x_train, s_train = sample_data(2048)
beta_i = get_beta_i(model.N)
train_loader = get_dataloader(
    x_train, s_train, beta_i, use_cuda=use_cuda, batch_size=16
)
x_valid, s_valid = sample_data(128)
valid_loader = get_dataloader(
    x_valid, s_valid, beta_i, use_cuda=use_cuda, batch_size=16
)
optimizer = Adam(model.parameters(), lr=1.0e-03)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=5,
)
manager = get_extension_manager(
    model,
    optimizer,
    train_loader,
    valid_loader,
    epochs=100,
    device=device,
    lr_scheduler=lr_scheduler,
    out_dir="result_gauss",
    snapshot=None,
)
logger.info(f"{device=}")
train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    manager=manager,
    device=device,
)
