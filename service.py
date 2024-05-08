import os
import uuid
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

from pack_model import BENTO_MODEL_TAG


MAX_TOKENS = 1024
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


@bentoml.service(
    name="bentomlcllm-llama3-8b-insruct-service",
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class MLCLLM:

    bento_model_ref = bentoml.models.get(BENTO_MODEL_TAG)

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from mlc_llm.serve import AsyncMLCEngine

        model_path = self.bento_model_ref.path
        self.engine = AsyncMLCEngine(
            model=model_path,
            model_lib=os.path.join(model_path, "lib.so"),
            mode="server"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.bento_model_ref.path)
        self.stop_tokens = [
            tokenizer.convert_ids_to_tokens(
                tokenizer.eos_token_id,
            ),
            "<|eot_id|>",
        ]


    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
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
