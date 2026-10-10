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

## Important current limitation

The packer converts the existing portable mobile export; it does **not** turn an arbitrary PyTorch checkpoint into a compatible model or make a weak model intelligent by compression alone. A model architecture/exporter change is required before a new cloud-trained architecture can run on the phone. The phone runtime also needs real-device profiling before claiming a latency or RAM target.
