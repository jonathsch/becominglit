# Data Preprocessing

Before training the avatar, we need to generate annotations for the input images. This involves several steps, including alpha masks, semantic segmentations, landmark detection and FLAME tracking.

## Environment variables
The project reads environment variables from `~/.config/becominglit/.env` by default (see `becominglit/util/env.py`). Relevant variables:

- `BECOMINGLIT_DATASET_PATH` – root path to the raw/working dataset.
- `BECOMINGLIT_FLAME_TRACKING_PATH` – path to FLAME tracking results (used by `flame_tracker.py`).

If any of the required variables are missing the code exits with an explanatory message.


## Steps to preprocess data
(command examples assume running from the repository root.)

1. **BirefNet**: Create foreground alpha masks
```bash
python -m becominglit.data.preprocess.birefnet $SID
```

2. **Facer**: Create semantic segmentations
```bash
python -m becominglit.data.preprocess.facer $SID
```

3. **PIPNet**: Detect facial landmarks
```bash
python -m becominglit.data.preprocess.pipnet $SID
```

4. **FLAME tracker**: Fit FLAME model to each sequence (this needs to be done separately for every sequence)
```bash
python -m becominglit.data.preprocess.flame_tracker --data.subject $SID --data.sequence $SEQUENCE_NAME
```

5. **Interpolation**: Since all annotations were only generated for the fully-lit frames, we need to interpolate all annotations to the OLAT frames. (this needs to be done separately for every sequence):
```bash
# Interpolate masks
python -m becominglit.data.preprocess.interpolate_masks $SID $SEQUENCE_NAME

# Export latest FLAME trackings to main dataset folder
python -m becominglit.data.preprocess.export $SID $SEQUENCE_NAME
```

Refer to each script's `--help` for exact CLI flags.

The dataset is now preprocessed and ready for training!

## Troubleshooting
- Alpha map generation and FLAME tracking are GPU memory intensive (We tested on an RTX3090 with 24GB VRAM). If you run out of memory, try lowering the image resolution.
