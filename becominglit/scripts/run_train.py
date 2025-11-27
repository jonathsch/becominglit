import logging
import sys
from copy import deepcopy
from pathlib import Path

# import lovely_tensors as lt
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from becominglit.data.bl_dataset import (
    collate_fn,
    worker_init_fn,
)
from becominglit.util import env
from becominglit.util.module_loader import build_optimizer, load_from_config
from becominglit.util.torchutils import load_checkpoint
from becominglit.util.train import train_loop

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(config: DictConfig):
    device = torch.device("cuda:0")

    train_dataset = load_from_config(config.data)
    val_dataset = load_from_config(config.val_data) if "val_data" in config else None
    assets = train_dataset.load_assets()
    batch_transform_fn = train_dataset.batch_transform_fn

    # albedo_init = assets["color_mean"]
    model = load_from_config(config.model, init_albedo=assets["mean_texture"]).to(device)
    summary_fn = load_from_config(config.summary)
    loss_fn = load_from_config(config.loss).to(device)

    optimizer = build_optimizer(config.optimizer, model)

    config.train.log_dir = env.BECOMINGLIT_EXPERIMENT_PATH
    Path(config.train.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(config.train.run_dir).joinpath("images").mkdir(exist_ok=True)
    start_iter = 0
    if "ckpt" in config.train:
        logger.info(f"Resuming from checkpoint: {config.train.ckpt}")
        ckpt_iter = load_checkpoint(
            config.train.ckpt,
            modules={
                "model": model,
                "optimizer": optimizer,
            },
        )
        start_iter = ckpt_iter + 1

    wandb_run = (
        wandb.init(
            project="becominglit",
            dir=config.train.run_dir,
            name=config.train.run_id,
            group=config.train.tag,
            tags=[config.train.tag],
            config=OmegaConf.to_container(config),
        )
        if config.train.run_id != "debug"
        else None
    )

    logger.info("Starting training with the config:")
    logger.info(OmegaConf.to_yaml(config))
    OmegaConf.save(config, f"{config.train.run_dir}/config.yml")

    # warmup phase (fully lit only)
    if start_iter < config.train.n_warmup_iters:
        warmup_data_config = deepcopy(config.data)
        warmup_data_config.fully_lit_only = True
        warmup_set = load_from_config(warmup_data_config)
        warmup_loader = DataLoader(
            warmup_set, collate_fn=collate_fn, worker_init_fn=worker_init_fn, **config.dataloader
        )
        n_warmup_iters = config.train.n_warmup_iters - start_iter
        logger.info(f"Starting warmup phase for {n_warmup_iters} iterations (fully lit only).")
        start_iter = train_loop(
            model,
            optimizer,
            warmup_loader,
            config,
            loss_fn,
            n_warmup_iters,
            wandb_run,
            batch_transform_fn,
            summary_fn,
            start_iter,
            device=device,
        )
        logger.info("Warmup phase completed.")

    # full training phase
    logger.info("Starting training.")
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, worker_init_fn=worker_init_fn, **config.dataloader)
    val_loader = (
        DataLoader(val_dataset, collate_fn=collate_fn, worker_init_fn=worker_init_fn, **config.dataloader)
        if val_dataset is not None
        else None
    )
    train_loop(
        model,
        optimizer,
        train_loader,
        config,
        loss_fn,
        config.train.n_max_iters,
        wandb_run,
        batch_transform_fn,
        summary_fn,
        start_iter,
        val_loader,
        device,
    )

    logger.info("Training completed.")


if __name__ == "__main__":
    config_path = sys.argv[1]
    cli_arguments = sys.argv[2:]

    config = OmegaConf.load(config_path)
    config_cli = OmegaConf.from_cli(args_list=cli_arguments)

    if config_cli:
        logger.info("Overriding config with CLI arguments")
        logger.info(f"{OmegaConf.to_yaml(config_cli)}")
        config = OmegaConf.merge(config, config_cli)

    main(config)
