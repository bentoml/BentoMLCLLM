## Installation

I suggest using python 3.11 (though 3.10 seems to work too).

```bash
python3.11 -m venv venv
source venv/bin/activate
# for cuda 12.1
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu121 mlc-ai-nightly-cu121
```


## Compile and pack the model

```bash
# For 8B model
git clone https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q0f16-MLC
export MLC_MODEL_PATH=Llama-3-8B-Instruct-q0f16-MLC

# For 70B model
git clone https://huggingface.co/mlc-ai/Llama-3-70B-Instruct-q4f16_1-MLC
export MLC_MODEL_PATH=Llama-3-70B-Instruct-q4f16_1-MLC

export MODEL_LIB=$MLC_MODEL_PATH/lib.so
mlc_llm compile $MLC_MODEL_PATH -o $MODEL_LIB --device cuda

# pack the model into model store
python3 pack_model.py

# build bento
bentoml build .
```


## Notes

- need to specify cuda version in bentofile because tvm need some cuda library
- remove setup.sh after bentoml fix requirements.txt bug
