import glob
import os
import shutil

import bentoml

MODEL_DIR = "Llama-3-8B-Instruct-q0f16-MLC"
BENTO_MODEL_TAG = MODEL_DIR.lower() + "-mlcllm" + "-cuda"


def pack_model(bento_model_tag, model_dir):
    with bentoml.models.create(bento_model_tag) as bento_model_ref:
        for filename in glob.glob(os.path.join(model_dir, "*.*")):
            shutil.copy(filename, bento_model_ref.path)

if __name__ == "__main__":
    pack_model(BENTO_MODEL_TAG, MODEL_DIR)
