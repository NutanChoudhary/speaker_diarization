"""
titanet_to_onnx.py
==================
Downloads TitaNet-Large from NVIDIA NeMo and saves it as ONNX.

If you run into any issues downloading the pre-converted `titanet.onnx` file, you can run the script below to download and export the model yourself. 

Before running the script, make sure you install the required NeMo ASR dependencies:


pip install "nemo_toolkit[asr]"

""""

import os
import nemo.collections.asr as nemo_asr

ONNX_OUTPUT_PATH = "titanet_large.onnx"
os.makedirs(os.path.dirname(ONNX_OUTPUT_PATH), exist_ok=True)

# 1. Download
print("Downloading TitaNet-Large from NeMo NGC...")
model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
    model_name="titanet_large"
)
model.eval()
print(f"Model loaded: {model.__class__.__name__}")

# 2. Export to ONNX
print(f"Exporting to {ONNX_OUTPUT_PATH} ...")
model.export(
    output=ONNX_OUTPUT_PATH,
    onnx_opset_version=17,
    check_trace=True,
    verbose=False,
)
print(f"✓ Saved at: {ONNX_OUTPUT_PATH}")
