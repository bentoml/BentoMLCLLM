import os
import uuid
from typing import AsyncGenerator, Optional

import bentoml
import fastapi
from annotated_types import Ge, Le
from typing_extensions import Annotated


MAX_TOKENS = 2048
MAX_NEW_TOKENS =1024

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

MODEL_URL = "HF://mlc-ai/Llama-3.1-8B-Instruct-q0f16-MLC"
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

openai_api_app = fastapi.FastAPI()

runtime_image = bentoml.images.PythonImage(python_version='3.11')\
                              .requirements_file('requirements.txt'),

@bentoml.asgi_app(openai_api_app, path="/")
@bentoml.service(
    name="bentomlcllm-llama3.1-8b-insruct-service",
    image=runtime_image,
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class MLCLLM:

    @bentoml.on_startup
    def init(self) -> None:
        from transformers import AutoTokenizer
        from mlc_llm.serve import engine
        from mlc_llm.serve.entrypoints import openai_entrypoints
        from mlc_llm.serve.server import ServerContext

        self.engine = engine.AsyncMLCEngine(
            model=MODEL_URL,
            mode="server",
            engine_config=engine.EngineConfig(
                max_single_sequence_length=MAX_NEW_TOKENS,
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.stop_tokens = [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.eos_token_id,
            ),
            "<|eot_id|>",
        ]

        self.server_ctx = ServerContext()
        self.server_ctx.__enter__()
        self.server_ctx.add_model(MODEL_ID, self.engine)

        openai_api_app.include_router(openai_entrypoints.app)


    @bentoml.on_shutdown
    def cleanup(self) -> None:
        self.server_ctx.__exit__(GeneratorExit, None, None)


    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_NEW_TOKENS,
    ) -> AsyncGenerator[str, None]:

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        session_id = uuid.uuid4().hex
        messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

        stream = await self.engine.chat.completions.create(
            messages=messages,
            request_id=session_id,
            max_tokens=max_tokens,
            stop=self.stop_tokens,
            stream=True,
        )

        async for request_output in stream:
            yield request_output.choices[0].delta.content
