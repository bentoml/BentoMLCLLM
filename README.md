<div align="center">
    <h1 align="center">Self-host LLMs with MLC-LLM and BentoML</h1>
</div>

This is a BentoML example project, showing you how to serve and deploy open-source Large Language Models using [MLC-LLM](https://github.com/mlc-ai/mlc-llm), a machine learning compiler and high-performance deployment engine for large language models.

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

💡 This example is served as a basis for advanced code customization, such as custom model, inference logic or LMDeploy options. For simple LLM hosting with OpenAI compatible endpoint without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

## Prerequisites

- You have installed Python 3.11+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- If you want to test the Service locally, you need a Nvidia GPU with at least 20G VRAM.
- This example uses Llama 3. Make sure you have [gained access to the model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Set up the environment

Clone the repo:

```bash
git clone https://github.com/bentoml/BentoMLCLLM.git
cd BentoMLCLLM
```

Install dependencies:

```bash
# for cuda 12.1
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu121 mlc-ai-nightly-cu121
pip install -r requirements.txt
```

Compile and pack the model:

```bash
# For 8B model
git clone https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q0f16-MLC
export MLC_MODEL_PATH=Llama-3-8B-Instruct-q0f16-MLC

# For 70B model
git clone https://huggingface.co/mlc-ai/Llama-3-70B-Instruct-q4f16_1-MLC
export MLC_MODEL_PATH=Llama-3-70B-Instruct-q4f16_1-MLC

export MODEL_LIB=$MLC_MODEL_PATH/lib.so
mlc_llm compile $MLC_MODEL_PATH -o $MODEL_LIB --device cuda
```

Expected output:

```bash
...
[2024-06-06 11:51:49] INFO pipeline.py:52: Compiling external modules
[2024-06-06 11:51:49] INFO pipeline.py:52: Compilation complete! Exporting to disk
[2024-06-06 11:51:58] INFO model_metadata.py:95: Total memory usage without KV cache:: 15628.51 MB (Parameters: 15316.51 MB. Temporary buffer: 312.00 MB)
[2024-06-06 11:51:58] INFO model_metadata.py:103: To reduce memory usage, tweak `prefill_chunk_size`, `context_window_size` and `sliding_window_size`
[2024-06-06 11:51:58] INFO compile.py:193: Generated: Llama-3-8B-Instruct-q0f16-MLC/lib.so
```

Import the model into the BentoML Model Store:

```bash
python3 pack_model.py
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .
2024-06-06T12:22:33+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:MLCLLM" listening on http://localhost:3000 (Press CTRL+C to quit)
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Explain superconductors like I'\''m five years old",
  "max_tokens": 1024
}'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response_generator = client.generate(
        prompt="Explain superconductors like I'm five years old",
        max_tokens=1024
    )
    for response in response_generator:
        print(response, end='')
```

</details>

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it. Note that you need to specify the CUDA version in `bentofile.yaml`.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
