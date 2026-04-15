import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from becominglit.data.bl_dataset import (
    BecomingLitDataset,
    collate_fn,
    worker_init_fn,
)
from becominglit.util import image
from becominglit.util.inspection import filter_inputs
from becominglit.util.module_loader import load_from_config
from becominglit.util.torchutils import load_checkpoint, to_device

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MOTION_KEYS = ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation"]


@torch.no_grad()
def main(config: DictConfig):
    device = torch.device("cuda:0")

    # Load appearance dataset (provides camera, lighting, identity params)
    data_config = deepcopy(config.val_data)
    data_config.cameras_subset = ["222200037"]
    data_config.fully_lit_only = True
    dataset = load_from_config(data_config)
    batch_transform_fn = dataset.batch_transform_fn

    # Load model
    model = load_from_config(config.model).to(device)
    load_checkpoint(config.train.ckpt_dir, modules={"model": model})
    model.eval()

    # Load driver dataset
    driver_config = deepcopy(config.val_data)
    driver_config.subject = config.driver_subject
    driver_config.sequences_subset = [config.driver_sequence]
    driver_config.cameras_subset = ["222200037"]
    driver_config.fully_lit_only = True
    driver_dataset = load_from_config(driver_config)

    # Extract identity params from the appearance subject
    identity_item = dataset[0]
    identity_shape = identity_item["flame_params"]["shape"]
    identity_static_offset = identity_item["flame_params"]["static_offset"]

    # Output directory
    output_dir = Path(config.train.run_dir).joinpath("reenact", f"{config.driver_subject}_{config.driver_sequence}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, worker_init_fn=worker_init_fn)

    for i, batch in enumerate(tqdm(dataloader, desc="Rendering cross-reenactment")):
        batch = to_device(batch, device)
        batch_transform_fn(batch)

        # Load driver motion params for this frame
        driver_seq, driver_timestep, _ = driver_dataset.items[i % len(driver_dataset.items)]
        driver_flame = driver_dataset.load_flame_params(driver_seq, driver_timestep)

        # Replace motion params with driver's, keep identity from appearance
        for key in MOTION_KEYS:
            batch["flame_params"][key] = driver_flame[key].unsqueeze(0).to(device)
        batch["flame_params"]["shape"] = identity_shape.unsqueeze(0).to(device)
        batch["flame_params"]["static_offset"] = identity_static_offset.unsqueeze(0).to(device)

        # Recompute flame_verts with the hybrid params
        flame_verts, _, _ = model.flame_mod.forward(batch["flame_params"])
        batch["flame_verts"] = flame_verts

        preds = model(**filter_inputs(batch, model))

        img = make_grid(image.linear2srgb(preds["rgb"]).clamp(0, 1)).permute(1, 2, 0).mul(255).byte().cpu().numpy()
        Image.fromarray(img).save(output_dir.joinpath(f"frame_{i:06d}.jpg"))

    logger.info(f"Reenactment results saved to {output_dir}")


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
