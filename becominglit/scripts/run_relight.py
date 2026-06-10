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
from becominglit.util.gaussian_mesh import render_gaussian_ellipsoids
from becominglit.util.lbs import batch_rodrigues
from becominglit.util.light_decorator import EnvLightSpinDecorator, PointLightPathDecorator
from becominglit.util.mesh import NVDiffRenderer
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

    # Renderer for the Gaussian-as-ellipsoid-mesh visualization. Use the OpenGL
    # backend (as the FLAME tracker does on these nodes): the CUDA rasterizer
    # requires resolutions divisible by 8, which the render size here is not.
    gauss_renderer = NVDiffRenderer(use_opengl=True, lighting_type="front").to(device)

    # One fixed random color per Gaussian, kept consistent across all frames.
    n_gauss = int(model.flame_mod.valid_mask.sum().item())
    gauss_colors = torch.rand(n_gauss, 3, generator=torch.Generator(device=device).manual_seed(0), device=device)

    @torch.no_grad()
    def gaussian_mesh_vis(batch, preds):
        """Render the posed Gaussians as a solid ellipsoid mesh from the input view.

        The model works in head-relative space, so the camera is composed with the
        FLAME head pose (rotation + translation), exactly as in the model forward.
        """
        valid = model.flame_mod.valid_mask.squeeze(-1)  # [U, V]
        flame_params = batch["flame_params"]
        bs = batch["Rt"].shape[0]
        head_pose = torch.eye(4, device=device)[None].repeat(bs, 1, 1)
        head_pose[:, :3, :3] = batch_rodrigues(flame_params["rotation"])
        head_pose[:, :3, 3] = flame_params["translation"]
        rt_headrel = batch["Rt"] @ head_pose
        h, w = preds["rgb"].shape[-2:]
        # Model works in mm; render in meters so the head is within the clip planes.
        scene_scale = 1e-3 if dataset.length_unit == "mm" else 1.0
        rgb = render_gaussian_ellipsoids(
            gauss_renderer,
            preds["means"][0][valid],
            preds["scales"][0][valid],
            preds["quats"][0][valid],
            rt_headrel,
            batch["K"],
            (h, w),
            opacity=preds["opacities"][0][valid],
            opacity_thresh=0.01,
            level=0,
            scene_scale=scene_scale,
            colors=gauss_colors,
        )
        return rgb.permute(0, 3, 1, 2)  # [1, 3, H, W], display-space [0, 1]

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

        rgb = image.linear2srgb(preds["rgb"]).clamp(0, 1)
        gauss = gaussian_mesh_vis(batch, preds)  # [1, 3, H, W], already display-space
        img = make_grid(torch.cat([rgb, gauss], dim=0)).permute(1, 2, 0).mul(255).byte().cpu().numpy()
        Image.fromarray(img).save(point_light_dir.joinpath(f"{i:06d}.jpg"))

    # envmap rendering
    assert "envmap_path" in config and Path(config.envmap_path).exists(), (
        "envmap_path must be specified in the config and exist."
    )

    env_scale = config.get("env_scale", 1.0)
    model_env = EnvLightSpinDecorator(model, config.envmap_path, assets, cycle=256, env_scale=env_scale).to(device)
    for i, batch in enumerate(tqdm(dataloader, desc="Rendering envmap frames")):
        batch = to_device(batch, device)
        batch_transform_fn(batch)

        preds = model_env(**batch, index=[i], render_auxiliary=True)

        rgb = image.linear2srgb(preds["rgb"]).clamp(0, 1)
        diffuse = image.linear2srgb(preds["render_diffuse"]).clamp(0, 1)
        specular = image.linear2srgb(preds["render_specular"]).clamp(0, 1)
        gauss = gaussian_mesh_vis(batch, preds)  # already display-space [0, 1]

        grid_img = make_grid(torch.cat([rgb, diffuse, specular, gauss], dim=0))
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
