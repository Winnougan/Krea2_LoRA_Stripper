# Krea2 LoRA Stripper
<img width="2666" height="2000" alt="Sample_01" src="https://github.com/user-attachments/assets/d12908b4-5b51-44af-8c83-d4d693f0294c" />
<img width="2666" height="2000" alt="Sample_03" src="https://github.com/user-attachments/assets/4cc066f5-f098-44c7-b32d-5ce0c141b24b" />
<img width="2666" height="2000" alt="Sample_02" src="https://github.com/user-attachments/assets/7f287c78-fb55-4020-892a-7584a03c286e" />

A small utility for shrinking **Krea2** LoRA/LoKr files by removing the DIT/UNET weights and keeping only the text-conditioning (`txtfusion`) layers.

The idea: for many style/character LoRAs, the base Krea2 model already "knows" how to render the relevant visual content — the LoRA's real job is steering the
text-conditioning path. Dropping the DIT/UNET block weights and keeping only the `txtfusion` layers can shrink files by ~90% with often minimal loss in output
fidelity for that class of LoRA.

> **This is not free lunch.** It works well for LoRAs that reinforce a style or
> concept the base model already has some notion of. It is *not* expected to
> work well for LoRAs that teach a genuinely novel subject, pose, or composition
> the base model has never seen — always compare outputs before trusting a
> stripped file for real use.

## Credit where it's due:
Puppet_Master on Civitai Red first posted the code on how he was able to reduce the file size by 90% of the Krea 2 lora size.
Source: https://civitai.red/models/2742336/nsfw-krea2-low-vram?modelVersionId=3089248

## How it works

Krea2 LoRA `.safetensors` files store weights under two broad key groups:

- `diffusion_model.blocks.N.*` — the DIT/UNET transformer blocks (bulk of file size)
- `diffusion_model.txtfusion.*` — text-conditioning fusion layers (layerwise
  attention blocks, refiner blocks, projector)
- `diffusion_model.txtmlp.*` — final text MLP/projection (not present in all LoRAs)

This tool removes the `blocks.*` (and a few related timestep/output) tensors and
keeps everything else, using the `safetensors` Python library. Original files are
never modified — a new `*_stripped.safetensors` file is written alongside the
original.

Only files whose keys include the `diffusion_model.txtfusion.` signature are
touched. Anything else (Flux, WAN, LTX, SDXL, etc. LoRAs — which use different
internal architectures) is detected and skipped automatically.

## Requirements

- Python 3.9+
- `safetensors` (`pip install safetensors`)
- PyTorch (`safetensors.torch` is used for loading/saving tensors)

## Usage

### Option 1 — one-click `.bat` (Windows)

1. Put `batch_strip_krea2.py` and `strip_krea2_loras.bat` in the same folder.
2. Double-click `strip_krea2_loras.bat`.
3. Paste the full path to your LoRA folder when prompted, hit Enter.
4. Stripped copies (`*_stripped.safetensors`) appear next to the originals.

### Option 2 — command line

```bash
python batch_strip_krea2.py "D:\path\to\your\lora\folder"
```

This scans every `.safetensors` file in the given folder and:

- Skips anything that isn't Krea2 architecture (no `txtfusion` keys found)
- Skips files that already have a corresponding `_stripped.safetensors` output
- Strips and writes a new file for everything else, preserving original metadata

### Dry run

To see what *would* happen without writing any files, open `batch_strip_krea2.py`
and set:

```python
DRY_RUN = True
```

## Example output

```
Found 3 .safetensors files in D:\ComfyUI\models\loras\Krea2

Checking: anya_krea2.safetensors
  -> Stripped: removed 448 tensors, kept 64
  -> 218.00 MB -> 13.20 MB (93.9% reduction)

Checking: some_flux_lora.safetensors
  -> SKIP (not Krea2 architecture / no txtfusion keys found)

Checking: anya_krea2_stripped.safetensors
  -> (skipped: already-stripped output file)

============================================================
SUMMARY
============================================================
[OK]      anya_krea2.safetensors: 218.0MB -> 13.2MB
[SKIP]    some_flux_lora.safetensors: not Krea2 architecture
```

## Recommended workflow

1. Run a dry run first to confirm which files will be affected.
2. Strip a small batch and A/B test in ComfyUI: same seed, same prompt, original LoRA vs. stripped LoRA.
3. If fidelity holds up for your use case, run the full batch.
4. Keep your original files — this tool never deletes or overwrites them, but it's still good practice to back up a LoRA library before batch-processing it.

## Limitations

- Krea2 architecture only. Will not do anything useful for other model families.
- Not all LoRAs compress equally well — low-rank or highly subject-specific LoRAs may lose meaningful fidelity when DIT weights are removed.
- This tool does not evaluate LoRA *content* — only its tensor architecture. Users are responsible for the legality and appropriateness of the files they
  process with it.

## Credit

Technique based on a size-reduction approach for Krea2 LoRAs shared on Civitai, from "Puppet_Master" using the `safetensors` Python library to selectively drop DIT/UNET tensors while retaining text-fusion layers.

💙Join my Ko-fi:
👉 https://ko-fi.com/winnougan
I'm also saving toward a hardware upgrade so I can expand into Klein9b, Ideogram 4 and LTX-2.3 video LoRAs and deliver releases faster. Every coffee helps make it happen. ☕

All LoRAs have all their epochs for members on Ko-fi.

Check out my YouTube channel for tutorials and reviews here:

💙Youtube:
🤘https://www.youtube.com/@Winnougan
💙 Discord:
Join my Discord, where I talk about all of my releases and entertain requests for future LoRAs!

👉 https://discord.gg/CJv5wceJaN
## License

MIT — use at your own risk. No warranty on output fidelity.
