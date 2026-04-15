import logging
import sys
from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision.utils import save_image
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


@torch.no_grad()
def main(config: DictConfig):
    device = torch.device("cuda:0")

    data_config = deepcopy(config.val_data)
    dataset = load_from_config(data_config)
    batch_transform_fn = dataset.batch_transform_fn

    model = load_from_config(config.model).to(device)
    load_checkpoint(
        config.train.ckpt_dir,
        modules={
            "model": model,
        },
    )
    model.eval()

    # make output folder
    output_dir = Path(config.train.run_dir).joinpath("test")
    output_dir.mkdir(parents=True, exist_ok=True)

    psnr_fn = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    for i, batch in enumerate(tqdm(dataloader, desc="Running test set")):
        batch = to_device(batch, device)
        batch_transform_fn(batch)

        frame = batch["frame"].item()
        serial = batch["serial"][0]

        preds = model(**filter_inputs(batch, model))

        gt_img = image.linear2srgb(batch["image"])
        pred_img = image.linear2srgb(preds["rgb"])
        error = ((gt_img - pred_img) ** 2) * 20.0

        psnr_fn.update(pred_img, gt_img)
        ssim_fn.update(pred_img, gt_img)
        lpips_fn.update(pred_img, gt_img)

        save_image(
            torch.cat([gt_img, pred_img, error], dim=-1),
            output_dir.joinpath(f"frame_{frame:06d}_cam_{serial}.jpg"),
            nrow=1, normalize=True, value_range=(0, 1),
        )

    psnr_score = psnr_fn.compute().item()
    ssim_score = ssim_fn.compute().item()
    lpips_score = lpips_fn.compute().item()

    logger.info(f"PSNR: {psnr_score:.4f}")
    logger.info(f"SSIM: {ssim_score:.4f}")
    logger.info(f"LPIPS: {lpips_score:.4f}")

    with open(output_dir.joinpath("metrics.txt"), "w") as f:
        f.write(f"PSNR: {psnr_score}\n")
        f.write(f"SSIM: {ssim_score}\n")
        f.write(f"LPIPS: {lpips_score}\n")


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
