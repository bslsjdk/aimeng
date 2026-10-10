# AIMG v1 portable mobile model format

AIMG is the canonical **delivery format** between cloud export and the Android inference runtime. Training checkpoints remain training-side artifacts; `.pth` is not loaded by the phone.

## Binary layout

All integers are unsigned-sized values represented as big-endian 32-bit integers.

| Offset | Length | Field |
|---:|---:|---|
| 0 | 4 | ASCII magic `AIMG` |
| 4 | 4 | Version, currently `1` |
| 8 | 4 | Uncompressed UTF-8 JSON payload length |
| 12 | 32 | SHA-256 digest of the uncompressed JSON bytes |
| 44 | remaining | GZIP-compressed UTF-8 JSON payload |

The decompressed JSON payload uses the existing `aimeng-mobile-diffusion-json-v1` schema with `config`, `stoi`, `itos`, and `tensors`. The binary wrapper standardizes delivery and verifies integrity; it does not alter model mathematics.

## Resource limits

- Packed bundle: at most 32 MiB.
- Decompressed JSON payload: at most 64 MiB.
- The app must validate the bundle before replacing the previous working model.
- Runtime pools must be budgeted and populated on demand. Do not allocate 2.5 GiB eagerly at startup.
- AIMG's file-size cap is not a promise that a model fits in RAM. JSON parsing temporarily needs additional memory, so true-device peak PSS must be measured.

## Pack from the cloud export

```bash
python scripts/pack_mobile_aimg.py mobile_diffusion.json mobile_diffusion.aimg
```

The packer checks the schema, writes a deterministic GZIP payload, writes the length and SHA-256 header, fsyncs a temporary file, and atomically replaces the destination.

## Migration and compatibility

The Android app continues to accept the legacy JSON bundle while `.aimg` becomes the preferred format. A future format version may store tensor tables and individual neuron shards directly, enabling true on-demand cold-neuron paging without parsing the entire JSON tensor map. That sharded runtime is a separate architectural step and must not be claimed as implemented until the forward pass actually loads neuron records from the external store.
