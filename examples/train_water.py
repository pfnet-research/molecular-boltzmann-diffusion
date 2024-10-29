import polars as pl

import torch
from torch.optim.adam import Adam
from loguru import logger

from moldiv.util import set_device_around, get_beta_i
from moldiv.molecule import (
    ScoreTransformer,
    get_dataloader,
    get_extension_manager,
    train,
)

df = pl.read_parquet("data/water/x_score_300.0K.parquet")
logger.info(df.describe)

use_cuda = torch.cuda.is_available()
device = set_device_around(use_cuda, seed=0)
model = ScoreTransformer().to(device)

n_train = len(df) * 9 // 10
df_train = df[:n_train]
x_train = df_train["positions"]
Z_train = df_train["numbers"]
s_train = df_train["score"]
x_train = torch.tensor(x_train).to(torch.float32)
Z_train = torch.tensor(Z_train).to(torch.int32)
s_train = torch.tensor(s_train).to(torch.float32)

df_valid = df[n_train:]
logger.debug(len(df_valid))
x_valid = df_valid["positions"]
Z_valid = df_valid["numbers"]
s_valid = df_valid["score"]
x_valid = torch.tensor(x_valid).to(torch.float32)
Z_valid = torch.tensor(Z_valid).to(torch.int32)
s_valid = torch.tensor(s_valid).to(torch.float32)

beta_i = get_beta_i(model.N)


train_loader = get_dataloader(
    x_train,
    Z_train,
    s_train,
    beta_i,
    use_cuda=use_cuda,
    batch_size=256,
    batch_size_time=4,
)
valid_loader = get_dataloader(
    x_valid,
    Z_valid,
    s_valid,
    beta_i,
    use_cuda=use_cuda,
    batch_size=256,
    batch_size_time=4,
)

optimizer = Adam(model.parameters(), lr=1.0e-03)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=3,
)

manager = get_extension_manager(
    model,
    optimizer,
    train_loader,
    valid_loader,
    epochs=60,
    device=device,
    lr_scheduler=lr_scheduler,
    out_dir="result_water",
    snapshot=None,
    weight_t1=0.0,
)
logger.info(f"{device=}")

train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    manager=manager,
    device=device,
    weight_t1=0.0,
)
