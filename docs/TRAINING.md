# Training VLA-JEPA

This guide explains how to train the VLA-JEPA model using the updated training script.

## Overview

The training script `starVLA/training/train_vlajepa_cotrain.py` has been updated to focus exclusively on training the VLA-JEPA model. The previous VLM co-training logic has been removed to streamline the process and ensure compatibility with the new C-JEPA architecture features (Slot Attention, Object Masking).

## Prerequisites

- **Environment**: Ensure your python environment is activated and dependencies are installed.
- **Data**: Ensure your dataset is prepared and accessible as configured in `starVLA/config/training/starvla_cotrain_oxe.yaml`.

## Configuration

The primary configuration file is:
`starVLA/config/training/starvla_cotrain_oxe.yaml`

Key VLA-JEPA parameters (recently added/verified):
- `num_slots`: Number of slots for Slot Attention (default: 8 or as set in config).
- `slot_dim`: Dimension of each slot.
- `num_iterations`: Number of Slot Attention iterations.
- `eps`: Epsilon for Slot Attention stability.

## Running Training

To start training, run the script as a python module using `accelerate launch`:

```bash
accelerate launch -m starVLA.training.train_vlajepa_cotrain \
    --config_yaml starVLA/config/training/starvla_cotrain_oxe.yaml
```

**Note**: You can override config parameters via command line, e.g.:
```bash
accelerate launch -m starVLA.training.train_vlajepa_cotrain \
    --config_yaml starVLA/config/training/starvla_cotrain_oxe.yaml \
    trainer.max_train_steps=10000
```

## Architecture Notes

- **Frozen Backbone**: The V-JEPA encoder is frozen during training. Only the Predictor, Action Head, and Slot Attention modules are trained.
- **Teacher Path**: The teacher encoder uses the same frozen V-JEPA backbone but with masked inputs (if configured) or full inputs for target generation, plus text features.
- **EMA**: Exponential Moving Average (EMA) for the teacher is **not required** as the backbone is frozen.

## Output

- **Checkpoints**: Saved in `runs/<run_id>/checkpoints`.
- **Logs**: Tensorboard logs are saved in `runs/<run_id>/tensorboard`.
