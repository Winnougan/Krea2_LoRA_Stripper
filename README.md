# Krea2 LoRA Stripper
<img width="2666" height="2000" alt="Sample_01" src="https://github.com/user-attachments/assets/d12908b4-5b51-44af-8c83-d4d693f0294c" />

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

What he originally wrote: 
Based on good NSFW Lora/LoKr, I've experimented with reduced size/weight: dropping the DIT/UNET layer part because the model already know how to do NSFW. keeping only the text encoder/fusion layers from the NSFW lora.

This technique can reduce Krea2 lora weight (reduce ANY lora/lokr weight)

exemple with this lora 218mo => 13.5mo

exemple with LoKr 1.46go => 40mo or even 20mo in some case

I simply use the safetensors python lib and keep only the txtfusion.layerwise_blocks and txtfusion.refiner layers from NSFW lora. Dropping the other layers as the model know how to draw nsfw stuff ;).

I wanted to experiment with the Krea2 model and understand two things:

If the model was trainned on NSFW (I think it was or it couldn't draw NSFW without UNET/DIT layer part)

Where is the NSFW influence in the model: txtfusion.layerwise_blocks seems to have 80% of the NSFW decision making and txtfusion.refiner 20% ish. txtfusion.projector is a small vector and I think it should not be touched too much (it merge the 12 big vector from qwen3vl) but doesn't directly influence NSFW.

Detail and code at the end.

detail of the token handling before DIT as I think I understand it (maybe wrong):

'''
                          Frozen Qwen3-VL-4B (Text Encoder)
                                   (hidden dim ~2560)
                                          |
                                          |  Select 12 specific layers
                                          |  [2,5,8,11,14,17,20,23,26,29,32,35]
                                          v
                               12 x (Seq_len, 2560) hidden states
                                          |
                                          |  Stack / Concat along layer dim
                                          v
                              [12, Seq_len, 2560]  ← multilayer input
                                          |
               +---------------------------------------------------+
               |               txtfusion.layerwise_blocks          |
               |                    (2 blocks, cross-layer attn)   |
               +---------------------------------------------------+
                                          |
               Block 0:
               - prenorm (RMSNorm scale [2560])
               - attn: wq/wk/wv/wo/gate [2560,2560] + qknorm
               - mlp: down[2560→6912] / gate[6912,2560] / up[6912,2560]
               - postnorm (RMSNorm scale [2560])
                                          |
               Block 1: (same structure as Block 0)
                                          |
                                          v
                     Fused layer-wise features [Seq_len, 2560]
                                          |
               +---------------------------------------------------+
               |              txtfusion.projector                  |
               |          Linear(12 → 1) weight [1, 12]            |
               +---------------------------------------------------+
                                          |
                                          v
                     Single fused representation per token
                     (or pooled/processed) [Seq_len, 2560]
                                          |
               +---------------------------------------------------+
               |            txtfusion.refiner_blocks               |
               |               (2 bidirectional transformer blocks)|
               +---------------------------------------------------+
                                          |
               Refiner Block 0:
               - prenorm (RMSNorm [2560])
               - attn: wq/wk/wv/wo/gate [2560,2560] + qknorm
               - mlp: down/gate/up (same as above)
               - postnorm
                                          |
               Refiner Block 1: (identical structure)
                                          |
                                          v
                 Refined bidirectional text features [Seq_len, 2560]
                                          |
               +---------------------------------------------------+
               |                    txtmlp                         |
               |             (final text MLP / projection)         |
               +---------------------------------------------------+
                                          |
               txtmlp.0.scale [2560]          ← residual / scaling
                                          |
               txtmlp.1: Linear [2560 → 6144] + bias
                                          |
               txtmlp.3: Linear [6144 → 6144] + bias
                                          |
                                          v
               Final conditioned text embeddings
                     (ready for cross-attention into DiT)
                                          |
                                          ↓
                       Injected into first DiT / unet_blocks
                 (via cross-attention in the main transformer)
Code for the layer stripping:

from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path

def strip_layers(input_file, output_file, remove_prefixes):
    tensors = {}

    with safe_open(input_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            if any(key.startswith(prefix) for prefix in remove_prefixes):
                print(f"Removing {key}")
                continue
            tensors[key] = f.get_tensor(key)
    save_file(tensors, output_file, metadata=metadata)

inputFile = "i:/ComfyUI_windows_portable/ComfyUI/models/loras/krea2/KNPV3_1.safetensors"
stripLayer = [
    # linked to the DIT/UNET
    'blocks', 'first', 'last_linear', 'last.linear'
    # liked to unet/DIT time MLP and timestep projection
    'tmlp_', 'tproj_1',
    # keep commented for NSFW
    #'txtfusion.layerwise_blocks', 'txtfusion_layerwise_blocks', # <--- full censor
    #'txtfusion.projector', 'txtfusion_projector',              # <-- zero impact
    #'txtfusion.refiner', 'txtfusion_refiner',                  # <-- tiny impact ==> full censor
]
prefix = 'diffusion_model.'
stripLayer = [prefix+x for x in stripLayer]
print('layer to strip: ', stripLayer)

outPutFile = Path(inputFile).name

strip_layers(
    inputFile,
    outPutFile,
    stripLayer
)

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
