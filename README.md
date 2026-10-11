# AIMENG

AIMENG separates **cloud-side training** from **phone-side inference**.

- Cloud training may use PyTorch or other training frameworks internally.
- The Android runtime must not depend on `.pth` checkpoints or import PyTorch.
- The canonical mobile delivery format is **AIMG v1**: a small binary header, a compressed portable `aimeng-mobile-diffusion-json-v1` payload, and SHA-256 integrity verification.
- The Android app supports old JSON bundles during migration. New releases should ship `.aimg`.

## Export to phone

First export the model using the existing portable JSON schema with top-level `format`, `config`, `stoi`, `itos`, and `tensors`. Then pack it:

```bash
python scripts/pack_mobile_aimg.py mobile_diffusion.json mobile_diffusion.aimg
```

Copy the resulting `.aimg` file to the phone and import it in the AIMENG runtime app. The Android runtime validates the header, version, decompressed size, SHA-256, and model dimensions before replacing a working model.

See [AIMG format specification](docs/AIMG_FORMAT.md).

## AIMENG-MC staged parkour curriculum (research prototype)

The experimental Minecraft-control branch adds a generated voxel-data pipeline and recurrent training loop:

- [Observation protocol](docs/AIMENG_MC_3D_OBSERVATION.md)
- [Curriculum and dataset generation](docs/AIMENG_MC_TRAINING_DATA.md)
- [Training workflow and memory budget](docs/AIMENG_MC_CURRICULUM_TRAINING.md)
- [Synthetic curriculum generator](scripts/generate_aimeng_mc_curriculum.py)
- [Recurrent policy trainer](scripts/train_aimeng_mc_curriculum.py)

The training order is walking → turning → obstacle avoidance → small jump → gap jump → rotating jump → super jump → double jump. GitHub Actions runs generator/trainer tests and a tiny walking smoke-train; use the manual **AIMENG-MC curriculum training** workflow for a real synthetic training run and downloadable experiment artifacts.

**Scope warning:** these are synthetic grid-world tasks and abstract jump macros. They are not a working Bedrock Add-On, not live Minecraft control, and not an AIMG-compatible Android model. Real game physics, input delivery, export, latency and peak memory remain unverified.

## Important current limitation

The packer converts the existing portable mobile export; it does **not** turn an arbitrary PyTorch checkpoint into a compatible model or make a weak model intelligent by compression alone. A model architecture/exporter change is required before a new cloud-trained architecture can run on the phone. The phone runtime also needs real-device profiling before claiming a latency or RAM target.
