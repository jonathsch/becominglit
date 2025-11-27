import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pillow_avif  # noqa
import torch
from dreifus.matrix import CameraCoordinateConvention, Pose, PoseType
from PIL import Image
from torch.utils.data import default_collate
from torchvision.transforms.functional import pil_to_tensor, to_tensor

from becominglit.data.preprocess.utils import get_fully_lit_frames, ibu68_index_into_wflw
from becominglit.util import env, image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BecomingLitTrackingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subject: str,
        sequence: str,
        align_cameras_to_axes: bool = False,
        camera_coord_conversion: str = "opencv->opengl",
        calibrated: bool = True,
        target_extrinsic_type: Literal["w2c", "c2w"] = "w2c",
        n_downsample_rgb: Optional[int] = 2,
        scale_factor: float = 1.0,
        background_color: Optional[Literal["white", "black"]] = "white",
        use_alpha_map: bool = True,
        use_landmark: bool = True,
        use_color_correction: bool = True,
        img_to_tensor: bool = False,
        batchify_all_views: bool = False,
    ):
        super().__init__()
        self.root_folder = Path(env.BECOMINGLIT_DATASET_PATH)
        self.subject = subject
        self.sequence = sequence
        self.align_cameras_to_axes = align_cameras_to_axes
        self.camera_coord_conversion = camera_coord_conversion
        self.calibrated = calibrated
        self.target_extrinsic_type = target_extrinsic_type
        self.n_downsample_rgb = n_downsample_rgb
        self.scale_factor = scale_factor
        self.background_color = background_color
        self.use_alpha_map = use_alpha_map
        self.use_landmark = use_landmark
        self.use_color_correction = use_color_correction
        self.img_to_tensor = img_to_tensor
        self.batchify_all_views = batchify_all_views

        # Define paths and camera parameters
        self.properties = self.define_properties()
        self.camera_params = self.load_camera_params()
        self.camera_ids = sorted(list(self.camera_params.keys()))
        self.ccms = self.load_color_calibration()

        # Find tracking frames
        self.timestep_ids = get_fully_lit_frames(self.subject, self.sequence)
        self.timestep_indices = list(range(len(self.timestep_ids)))

        self.items: List[Dict[str, Any]] = []
        for frame_idx, frame_id in enumerate(self.timestep_ids):
            for cam_idx, cam_id in enumerate(self.camera_ids):
                self.items.append(
                    {
                        "timestep_index": frame_idx,
                        "timestep_id": frame_id,
                        "camera_index": cam_idx,
                        "camera_id": cam_id,
                    }
                )

    @property
    def num_timesteps(self):
        return len(self.timestep_indices)

    @property
    def num_cameras(self):
        return len(self.camera_ids)

    @property
    def seq_folder(self):
        return self.root_folder.joinpath(self.subject, "sequences", self.sequence)

    def __len__(self):
        if self.batchify_all_views:
            return self.num_timesteps
        else:
            return len(self.items)

    def __getitem__(self, idx: int):
        if self.batchify_all_views:
            return self.getitem_by_timestep(idx)
        else:
            return self.getitem_single_image(idx)

    def getitem_single_image(self, item_idx: int):
        item = deepcopy(self.items[item_idx])

        timestep_id = item["timestep_id"]
        cam_id = item["camera_id"]
        cam_prefix = self.properties["rgb"]["cam_id_prefix"]
        img_suffix = self.properties["rgb"]["suffix"]
        folder = self.properties["rgb"]["folder"]
        rgb_path = self.seq_folder / folder / f"{cam_prefix}{cam_id}" / f"frame_{timestep_id:06d}.{img_suffix}"
        item["rgb"] = np.array(Image.open(rgb_path))

        cam_param = self.camera_params[item["camera_id"]]
        item["intrinsic"] = cam_param["intrinsic"].clone()
        item["extrinsic"] = cam_param["extrinsic"].clone()

        # Load alpha map
        if self.use_alpha_map:
            alpha_folder = self.properties["alpha_map"]["folder"]
            alpha_suffix = self.properties["alpha_map"]["suffix"]
            alpha_path = (
                self.seq_folder / alpha_folder / f"{cam_prefix}{cam_id}" / f"frame_{timestep_id:06d}.{alpha_suffix}"
            )
            alpha_map = Image.open(alpha_path)
            item["alpha_map"] = to_tensor(alpha_map)

            seg_folder = self.properties["seg_masks"]["folder"]
            seg_suffix = self.properties["seg_masks"]["suffix"]
            seg_path = self.seq_folder / seg_folder / f"{cam_prefix}{cam_id}" / f"frame_{timestep_id:06d}.{seg_suffix}"
            seg_mask = Image.open(seg_path)
            item["seg_mask"] = pil_to_tensor(seg_mask)

        if self.use_landmark:
            lmks_path = self.seq_folder.joinpath("landmark2d", "PIPnet", f"cam_{cam_id}.npz")
            lmks_dict = np.load(lmks_path)

            if str(timestep_id) not in lmks_dict:
                item["lmk2d"] = torch.zeros(68, 3, dtype=torch.float32)
            else:
                # Convert from WFLW to IBU68 format
                lmks = lmks_dict[str(timestep_id)][ibu68_index_into_wflw]  # [N, 2]
                lmks = np.concatenate([lmks, np.ones_like(lmks[:, :1])], axis=-1)  # [N, 3]
                item["lmk2d"] = lmks

            if (item["lmk2d"][:, :2] == -1).sum() > 0:
                item["lmk2d"][:, 2:] = 0.0
            else:
                item["lmk2d"][:, 2:] = 1.0

        self.apply_scale_factor(item)
        if self.img_to_tensor:
            item["rgb"] = torch.as_tensor(item["rgb"], dtype=torch.float32)

        return item

    def batch_filter_fn(self, batch: Dict[str, Any]) -> None:
        """
        Combines all compute-intensive processing steps for when the batch is on the GPU.
        """
        device = batch["rgb"].device
        if batch["rgb"].dtype != torch.float32:
            batch["rgb"] = batch["rgb"].float()  # Ensure RGB images are float32
        batch["rgb"] /= 255.0  # Normalize RGB images to [0, 1]

        # Downsample RGB images if required
        if self.n_downsample_rgb:
            h, w = batch["rgb"].shape[-3:-1]  # [B, H, W, C]
            h //= self.n_downsample_rgb
            w //= self.n_downsample_rgb
            batch["rgb"] = image.scale_img_nhwc(batch["rgb"], (h, w))

            # Also downsample alpha map and segmenation mask if they exist
            if "alpha_map" in batch:
                batch["alpha_map"] = image.scale_img_nchw(batch["alpha_map"], (h, w))
            if "seg_mask" in batch:
                batch["seg_mask"] = image.scale_img_nchw(batch["seg_mask"], (h, w), mag="nearest", min="nearest")

        batch["rgb"] = batch["rgb"].permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Apply color correction
        if not isinstance(batch["camera_id"][0], str):
            # Flatten the camera_id list to match the batch size
            batch["camera_id"] = batch["camera_id"][0]
        if self.use_color_correction:
            ccm_mtx = torch.stack([self.ccms[cam_id] for cam_id in batch["camera_id"]], dim=0).to(device)  # [B, 3, 3]
            batch["rgb"] = image.color_correct_srgb(batch["rgb"], ccm_mtx)

        if self.background_color is not None:
            # Apply background color
            assert "alpha_map" in batch, "'alpha_map' is required to apply background color."
            fg = batch["rgb"]
            if self.background_color == "white":
                bg = torch.ones_like(fg)
            elif self.background_color == "black":
                bg = torch.zeros_like(fg)
            else:
                raise NotImplementedError(f"Unknown background color: {self.background_color}.")

            alpha = batch["alpha_map"]
            img = alpha * fg + (1 - alpha) * bg

            # Apply segmentation (mask out cloth and neck)
            if "seg_mask" in batch:
                seg_mask = batch["seg_mask"]
                is_cloth_or_neck = seg_mask == 3
                is_cloth_or_neck = is_cloth_or_neck.expand(-1, 3, -1, -1)
                img[is_cloth_or_neck] = bg[is_cloth_or_neck]

            batch["rgb"] = img

    def getitem_by_timestep(self, timestep_idx: int):
        begin = timestep_idx * self.num_cameras
        indices = range(begin, begin + self.num_cameras)
        item = default_collate([self.getitem_single_image(idx) for idx in indices])

        item["num_cameras"] = self.num_cameras
        return item

    def apply_scale_factor(self, item: Dict[str, Any]):
        # Downsample RGB images if required
        h, w = item["rgb"].shape[:2]
        if self.n_downsample_rgb:
            h //= self.n_downsample_rgb
            w //= self.n_downsample_rgb

        # properties that are defined based on image size
        if "lmk2d" in item:
            item["lmk2d"][..., 0] *= w
            item["lmk2d"][..., 1] *= h

        if "lmk2d_iris" in item:
            item["lmk2d_iris"][..., 0] *= w
            item["lmk2d_iris"][..., 1] *= h

        if "bbox_2d" in item:
            item["bbox_2d"][[0, 2]] *= w
            item["bbox_2d"][[1, 3]] *= h

    def define_properties(self):
        return {
            "rgb": {
                "folder": "images",
                "cam_id_prefix": "cam_",
                "per_timestep": True,
                "suffix": env.IMAGE_FILE_FORMAT,
            },
            "alpha_map": {
                "folder": "alpha_birefnet",
                "cam_id_prefix": "cam_",
                "per_timestep": True,
                "suffix": "jpg",
            },
            "seg_masks": {
                "folder": "facer_segmentation",
                "cam_id_prefix": "cam_",
                "per_timestep": True,
                "suffix": "png",
            },
            "landmark2d/face-alignment": {
                "folder": "landmark2d/face-alignment",
                "per_timestep": False,
                "suffix": "npz",
            },
            "landmark2d/STAR": {
                "folder": "landmark2d/STAR",
                "cam_id_prefix": "cam_",
                "per_timestep": False,
                "suffix": "npz",
            },
        }

    def get_property_path(
        self,
        name,
        index: Optional[int] = None,
        timestep_id: Optional[str] = None,
        camera_id: Optional[str] = None,
    ):
        p = self.properties[name]
        folder = p["folder"] if "folder" in p else None
        per_timestep = p["per_timestep"]
        suffix = p["suffix"]

        path = self.seq_folder
        if folder is not None:
            path = path / folder

        if self.num_cameras > 1:
            if camera_id is None:
                assert index is not None, "index is required when camera_id is not provided."
                camera_id = self.items[index]["camera_id"]
            if "cam_id_prefix" in p:
                camera_id = p["cam_id_prefix"] + camera_id
        else:
            camera_id = ""

        if per_timestep:
            if timestep_id is None:
                assert index is not None, "index is required when timestep_id is not provided."
                timestep_id = self.items[index]["timestep_id"]
            if len(camera_id) > 0:
                path /= f"{camera_id}_{timestep_id}.{suffix}"
            else:
                path /= f"{timestep_id}.{suffix}"
        else:
            if len(camera_id) > 0:
                path /= f"{camera_id}.{suffix}"
            else:
                path = Path(str(path) + f".{suffix}")

        return path

    def load_camera_params(self) -> Dict[str, Dict[str, torch.Tensor]]:
        cam_params_path = self.root_folder / self.subject / "calibration" / "camera_calibration.json"
        with open(cam_params_path, mode="r") as f:
            cam_calib = json.load(f)

        # Intrinsics
        cam_params = cam_calib["cam_data"]
        fx = cam_params["fx"] / self.n_downsample_rgb if self.n_downsample_rgb else cam_params["fx"]
        fy = cam_params["fy"] / self.n_downsample_rgb if self.n_downsample_rgb else cam_params["fy"]
        cx = cam_params["cx"] / self.n_downsample_rgb if self.n_downsample_rgb else cam_params["cx"]
        cy = cam_params["cy"] / self.n_downsample_rgb if self.n_downsample_rgb else cam_params["cy"]
        K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Extrinsics
        world_to_cams = {cid: np.array(w2c) for cid, w2c in cam_calib["world_to_cam"].items()}

        # Convert from OpenCV to OpenGL convention
        if self.camera_coord_conversion == "opencv->opengl":
            world_to_cams = {
                cid: Pose(
                    w2c, pose_type=PoseType.WORLD_2_CAM, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV
                )
                .change_pose_type(PoseType.CAM_2_WORLD)
                .change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)
                .change_pose_type(PoseType.WORLD_2_CAM)
                .numpy()
                for cid, w2c in world_to_cams.items()
            }

        return {
            cid: {"intrinsic": K, "extrinsic": torch.as_tensor(w2c, dtype=torch.float32)}
            for cid, w2c in world_to_cams.items()
        }

    def load_color_calibration(self) -> Dict[str, torch.Tensor]:
        ccm_path = self.root_folder.joinpath(self.subject, "calibration", "color_calibration.json")
        with open(ccm_path, mode="r") as f:
            ccm_data = json.load(f)

        ccm_dict = {}
        for cam_id, ccm in ccm_data.items():
            ccm_tensor = torch.tensor(ccm, dtype=torch.float32)
            ccm_dict[cam_id] = ccm_tensor

        return ccm_dict
