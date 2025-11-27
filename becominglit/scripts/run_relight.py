import logging
import sys
from copy import deepcopy
from pathlib import Path

import lovely_tensors as lt
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from becominglit.data.bl_dataset import (
    collate_fn,
    worker_init_fn,
)
from becominglit.util import image
from becominglit.util.light_decorator import EnvLightSpinDecorator, PointLightPathDecorator
from becominglit.util.module_loader import load_from_config
from becominglit.util.torchutils import load_checkpoint, to_device

lt.monkey_patch()

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def main(config: DictConfig):
    device = torch.device("cuda:0")

    data_config = deepcopy(config.val_data)
    data_config.cameras_subset = ["222200037"]
    data_config.fully_lit_only = True
    dataset = load_from_config(data_config)
    assets = dataset.load_assets()
    batch_transform_fn = dataset.batch_transform_fn

    model = load_from_config(config.model)

    load_checkpoint(
        config.train.ckpt_dir,
        modules={
            "model": model,
        },
    )

    # output folders
    output_dir = Path(config.train.run_dir).joinpath("relight")
    point_light_dir = output_dir.joinpath("point_light")
    point_light_dir.mkdir(parents=True, exist_ok=True)
    env_light_dir = output_dir.joinpath(f"env_{Path(config.envmap_path).stem}")
    env_light_dir.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, worker_init_fn=worker_init_fn)

    # point light rendering
    model_point = PointLightPathDecorator(model, cycle=256, light_rotate_axis=1).to(device)
    for i, batch in enumerate(tqdm(dataloader, desc="Rendering point light frames")):
        batch = to_device(batch, device)
        batch_transform_fn(batch)

        preds = model_point(**batch, index=[i], render_auxiliary=True)

        img = make_grid(image.linear2srgb(preds["rgb"]).clamp(0, 1)).permute(1, 2, 0).mul(255).byte().cpu().numpy()
        Image.fromarray(img).save(point_light_dir.joinpath(f"{i:06d}.jpg"))

    # envmap rendering
    assert "envmap_path" in config and Path(config.envmap_path).exists(), (
        "envmap_path must be specified in the config and exist."
    )

    model_env = EnvLightSpinDecorator(model, config.envmap_path, assets, cycle=256, env_scale=8.0).to(device)
    for i, batch in enumerate(tqdm(dataloader, desc="Rendering envmap frames")):
        batch = to_device(batch, device)
        batch_transform_fn(batch)

        preds = model_env(**batch, index=[i], render_auxiliary=True)

        rgb = preds["rgb"]
        diffuse = preds["render_diffuse"]
        specular = preds["render_specular"]

        grid_img = image.linear2srgb(make_grid(torch.cat([rgb, diffuse, specular], dim=0))).clamp(0, 1)
        grid_img = grid_img.permute(1, 2, 0).mul(255).byte().cpu().numpy()
        Image.fromarray(grid_img).save(env_light_dir.joinpath(f"{batch['frame'].item():06d}.jpg"))


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
