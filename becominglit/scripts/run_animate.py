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

    # Load appearance dataset for reference camera/lighting and identity params
    data_config = deepcopy(config.val_data)
    data_config.cameras_subset = ["222200037"]
    data_config.fully_lit_only = True
    dataset = load_from_config(data_config)
    batch_transform_fn = dataset.batch_transform_fn

    # Load model
    model = load_from_config(config.model).to(device)
    load_checkpoint(config.train.ckpt_dir, modules={"model": model})
    model.eval()

    # Extract reference camera/lighting from a single batch
    ref_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    ref_batch = to_device(next(iter(ref_loader)), device)
    batch_transform_fn(ref_batch)

    ref_Rt = ref_batch["Rt"]
    ref_K = ref_batch["K"]
    ref_width = ref_batch["width"]
    ref_height = ref_batch["height"]
    ref_light_pos = ref_batch["light_pos"]
    ref_light_intensity = ref_batch["light_intensity"]
    ref_n_lights = ref_batch["n_lights"]
    ref_is_fullylit = ref_batch["is_fullylit_frame"]

    # Extract identity params from the appearance subject
    identity_shape = ref_batch["flame_params"]["shape"]
    identity_static_offset = ref_batch["flame_params"]["static_offset"]

    # Load external FLAME parameters
    flame_params_path = Path(config.flame_params_path)
    assert flame_params_path.exists(), f"FLAME params file not found: {flame_params_path}"
    flame_data = np.load(flame_params_path)
    n_frames = flame_data["expr"].shape[0]
    logger.info(f"Loaded {n_frames} frames of FLAME parameters from {flame_params_path}")

    # Output directory
    output_dir = Path(config.train.run_dir).joinpath("animate", flame_params_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n_frames), desc="Rendering animation"):
        # Build flame_params: identity from dataset + motion from NPZ
        flame_params = {
            "shape": identity_shape,
            "static_offset": identity_static_offset,
        }
        for key in MOTION_KEYS:
            flame_params[key] = torch.as_tensor(flame_data[key][i], dtype=torch.float32).unsqueeze(0).to(device)

        # Compute flame_verts with the combined params
        flame_verts, _, _ = model.flame_mod.forward(flame_params)

        batch = {
            "flame_params": flame_params,
            "flame_verts": flame_verts,
            "Rt": ref_Rt,
            "K": ref_K,
            "width": ref_width,
            "height": ref_height,
            "light_pos": ref_light_pos,
            "light_intensity": ref_light_intensity,
            "n_lights": ref_n_lights,
            "is_fullylit_frame": ref_is_fullylit,
        }

        preds = model(**filter_inputs(batch, model))

        img = make_grid(image.linear2srgb(preds["rgb"]).clamp(0, 1)).permute(1, 2, 0).mul(255).byte().cpu().numpy()
        Image.fromarray(img).save(output_dir.joinpath(f"frame_{i:06d}.jpg"))

    logger.info(f"Animation results saved to {output_dir}")


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
