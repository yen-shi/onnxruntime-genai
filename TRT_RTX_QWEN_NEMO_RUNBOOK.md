# Qwen3.5 and Nemotron TRT-RTX Runbook

This runbook captures the commands needed to export and run:

- Qwen3.5 0.8B full VLM as `embedding.onnx`, `vision.onnx`, and `text.onnx`
- NVIDIA Nemotron 3.5 ASR streaming 0.6B as `encoder.onnx`, `decoder.onnx`, and `joint.onnx`
- NVIDIA Nemotron Parse v1.1 as `encoder.onnx` and `decoder.onnx`

The TRT-RTX examples assume a package folder that contains `onnxruntime_providers_nv_tensorrt_rtx.dll` and the matching TRT-RTX runtime DLLs.

## Common Setup

```powershell
$REPO = "C:\Users\yenshiw\work\trt"
$OGA = "$REPO\onnxruntime-genai"
$TRTRTX = "C:\path\to\trt-rtx-ep-package"
$EP_DLL = "$TRTRTX\onnxruntime_providers_nv_tensorrt_rtx.dll"

$env:PATH = "$TRTRTX;$env:PATH"
python -m pip install --upgrade pip
```

Use an onnxruntime-genai wheel built from this checkout, or run from the source tree when exporting:

```powershell
cd $OGA
python -m pip install -r requirements.txt
```

## Qwen3.5 0.8B Full VLM INT4 QDQ

The full VLM package is produced by the Olive recipe. It creates the support graphs and updates `genai_config.json` so GenAI can load the package as one model folder.

Expected output files:

```text
embedding.onnx
embedding.onnx.data
vision.onnx
vision.onnx.data
text.onnx
text.onnx.data
genai_config.json
processor_config.json
tokenizer.json
tokenizer_config.json
```

### Export for TRT-RTX EP

```powershell
$RECIPE = "$REPO\olive-recipes\Qwen-Qwen3.5-0.8B\builtin"
$QWEN_TRTRTX_OUT = "$REPO\_out\qwen35_08b_full_vlm_int4_qdq_trtrtx\models"

cd $RECIPE
python optimize.py --config-dir trtrtx --device trtrtx

New-Item -ItemType Directory -Force -Path $QWEN_TRTRTX_OUT | Out-Null
Copy-Item -Path ".\trtrtx\models\*" -Destination $QWEN_TRTRTX_OUT -Recurse -Force
```

The text decoder pass inside the recipe runs the builder with INT4 QDQ:

```powershell
python -m onnxruntime_genai.models.builder `
  -m Qwen/Qwen3.5-0.8B `
  -o trtrtx\models `
  -p int4 `
  -e NvTensorRtRtx `
  --extra_options `
    filename=text.onnx `
    quant_mode=int4 `
    use_qdq=true `
    int4_block_size=32 `
    int4_accuracy_level=4 `
    exclude_embeds=true `
    enable_cuda_graph=true
```

### Export for CUDA EP

```powershell
$RECIPE = "$REPO\olive-recipes\Qwen-Qwen3.5-0.8B\builtin"
$QWEN_CUDA_OUT = "$REPO\_out\qwen35_08b_full_vlm_int4_qdq_cuda\models"

cd $RECIPE
python optimize.py --config-dir cuda --device cuda

New-Item -ItemType Directory -Force -Path $QWEN_CUDA_OUT | Out-Null
Copy-Item -Path ".\cuda\models\*" -Destination $QWEN_CUDA_OUT -Recurse -Force
```

### Run Inference

TRT-RTX EP:

```powershell
cd "$OGA\examples\python"
python .\model-mm.py `
  -m $QWEN_TRTRTX_OUT `
  -e NvTensorRTRTXExecutionProvider `
  --ep_path $EP_DLL `
  --image_paths C:\path\to\image.jpg `
  --non_interactive `
  -up "Describe the image." `
  -g
```

CUDA EP:

```powershell
cd "$OGA\examples\python"
python .\model-mm.py `
  -m $QWEN_CUDA_OUT `
  -e cuda `
  --image_paths C:\path\to\image.jpg `
  --non_interactive `
  -up "Describe the image." `
  -g
```

TRT-RTX validation notes:

- `embedding.onnx` needs the TRT-RTX NonZero plug-in support.
- `vision.onnx` needs the TRT parser WinML `MultiHeadAttention` path that lowers to TRT Attention.
- `text.onnx` is the INT4 QDQ decoder graph.

## Nemotron 3.5 ASR Streaming 0.6B

Nemotron ASR is exported with the NeMo-specific export script. The generic LLM builder is not used for this model family.

Expected output files:

```text
encoder.onnx
encoder.onnx.data
decoder.onnx
decoder.onnx.data
joint.onnx
joint.onnx.data
genai_config.json
audio_processor_config.json
```

### Export

```powershell
$ASR_RECIPE = "$REPO\olive-recipes\nvidia-nemotron-speech-streaming-en-0.6b\scripts"
$ASR_OUT = "$REPO\_out\nemotron35_asr_export\onnx_models"

cd $ASR_RECIPE
python .\export_nemotron_to_onnx_static_shape.py `
  --model_name nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b `
  --output_dir $ASR_OUT `
  --streaming `
  --device cuda

python .\export_tokenizer.py --output_dir $ASR_OUT
```

If you are using the English checkpoint, replace `--model_name` with:

```text
nvidia/nemotron-speech-streaming-en-0.6b
```

### Run Inference

TRT-RTX EP:

```powershell
cd "$OGA\examples\python"
python .\nemotron_speech.py `
  --model_path $ASR_OUT `
  --audio_file C:\path\to\audio.wav `
  -e NvTensorRTRTXExecutionProvider `
  --ep_path $EP_DLL `
  --language en
```

CUDA EP:

```powershell
cd "$OGA\examples\python"
python .\nemotron_speech.py `
  --model_path $ASR_OUT `
  --audio_file C:\path\to\audio.wav `
  -e cuda `
  --language en
```

## Nemotron Parse v1.1

Nemotron Parse is exported as two component graphs:

```text
encoder.onnx
encoder.onnx.data
decoder.onnx
decoder.onnx.data
genai_config.json
```

The sample runner executes the component ONNX graphs directly through ONNX Runtime. This is enough for TRT-RTX and CUDA EP validation without adding a C++ GenAI runtime model type.

### Export for TRT-RTX EP

```powershell
$PARSE_TRTRTX_OUT = "$REPO\_out\nemotron_parse_trtrtx_int4_qdq"

cd $OGA
python -m onnxruntime_genai.models.builder `
  -m nvidia/NVIDIA-Nemotron-Parse-v1.1 `
  -o $PARSE_TRTRTX_OUT `
  -p int4 `
  -e NvTensorRtRtx `
  --extra_options `
    hf_token=true `
    hf_remote=true `
    use_qdq=true `
    int4_block_size=32 `
    int4_accuracy_level=4 `
    image_height=768 `
    image_width=768 `
    decoder_sequence_length=8 `
    export_components=encoder,decoder
```

### Export for CUDA EP

```powershell
$PARSE_CUDA_OUT = "$REPO\_out\nemotron_parse_cuda_int4_qdq"

cd $OGA
python -m onnxruntime_genai.models.builder `
  -m nvidia/NVIDIA-Nemotron-Parse-v1.1 `
  -o $PARSE_CUDA_OUT `
  -p int4 `
  -e cuda `
  --extra_options `
    hf_token=true `
    hf_remote=true `
    use_qdq=true `
    int4_block_size=32 `
    int4_accuracy_level=4 `
    image_height=768 `
    image_width=768 `
    decoder_sequence_length=8 `
    export_components=encoder,decoder
```

### Run Inference

TRT-RTX EP:

```powershell
cd "$OGA\examples\python"
python .\nemotron_parse.py `
  --model_path $PARSE_TRTRTX_OUT `
  --image_file C:\path\to\document.png `
  -e NvTensorRTRTXExecutionProvider `
  --ep_path $EP_DLL `
  --cache_dir "$PARSE_TRTRTX_OUT\trtrtx_cache" `
  --max_new_tokens 512
```

CUDA EP:

```powershell
cd "$OGA\examples\python"
python .\nemotron_parse.py `
  --model_path $PARSE_CUDA_OUT `
  --image_file C:\path\to\document.png `
  -e cuda `
  --max_new_tokens 512
```

Nemotron Parse export notes:

- `image_height` and `image_width` specialize the encoder graph resolution.
- `decoder_sequence_length` only controls the dummy input length used during export; the decoder graph keeps the decoder sequence axis dynamic.
- `export_components` can be set to `encoder`, `decoder`, or `encoder,decoder` during debug.
