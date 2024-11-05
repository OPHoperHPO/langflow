from pydantic.v1 import SecretStr

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.inputs import DropdownInput, FloatInput, IntInput, SecretStrInput, MultilineSecretInput, StrInput
from langflow.inputs.inputs import HandleInput
from __future__ import annotations

import asyncio
import time
from typing import List, Optional, Any, Callable

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
    _genai_extension as genaix,
)
from langchain_google_genai._common import (
    get_client_info,
)


class ModLLM(ChatGoogleGenerativeAI):

    def __init__(self, *args: Any, _token_factory: Callable[[], str], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._token_factory = _token_factory

    def _generate(self, *args, **kwargs):
        self.client = genaix.build_generative_service(
            credentials=self.credentials,
            api_key=self._token_factory(),
            client_info=get_client_info("ChatGoogleGenerativeAI"),
            client_options=self.client_options,
            transport=self.transport,
        )
        retry = 0
        while True:
            try:
                return super()._generate(*args, **kwargs)
            except Exception as e:
                print(e)
                retry += 1
                time.sleep(1)
                if retry > 10:
                    raise e
                self.client = genaix.build_generative_service(
                    credentials=self.credentials,
                    api_key=self._token_factory(),
                    client_info=get_client_info("ChatGoogleGenerativeAI"),
                    client_options=self.client_options,
                    transport=self.transport,
                )
                continue

    async def _agenerate(self, *args, **kwargs):
        self.async_client = genaix.build_generative_async_service(
            credentials=self.credentials,
            api_key=self._token_factory(),
            client_info=get_client_info("ChatGoogleGenerativeAI"),
            client_options=self.client_options,
            transport=self.transport,
        )
        retry = 0
        while True:
            try:
                return await super()._agenerate(*args, **kwargs)
            except Exception as e:
                print(e)
                await asyncio.sleep(1)
                retry += 1
                if retry > 10:
                    raise e

                self.async_client = genaix.build_generative_async_service(
                    credentials=self.credentials,
                    api_key=self._token_factory(),
                    client_info=get_client_info("ChatGoogleGenerativeAI"),
                    client_options=self.client_options,
                    transport=self.transport,
                )
                continue


class GeminiFactory:
    def __init__(
        self,
        api_keys: List[str],
        auth_postfix: Optional[str],
        base_url: Optional[str],
        transport: Optional[str] = "rest",
    ):
        self._api_keys = api_keys
        self._usage_per_key = [0] * len(api_keys)
        self._base_url = base_url
        self._transport = transport
        self._auth_postfix = auth_postfix

    def get_key(self) -> str:
        min_usage = min(self._usage_per_key)
        key_index = self._usage_per_key.index(min_usage)
        self._usage_per_key[key_index] += 1
        if self._usage_per_key[key_index] % 100 == 0:
            self._usage_per_key[key_index] = 0
        return self._api_keys[key_index] + self._auth_postfix

    def __call__(
        self,
        **kwargs,
    ) -> ModLLM:
        if not "safety_settings" in kwargs:
            kwargs["safety_settings"] = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }

        return ModLLM(
            transport=self._transport,
            client_options={"api_endpoint": self._base_url},
            google_api_key=self.get_key(),
            _token_factory=self.get_key,
            **kwargs,
        )

class GoogleGenerativeAIFactoryComponent(LCModelComponent):
    display_name = "Google Generative AI Factory"
    description = "Generate text using Google Generative AI. Supports Token Hotswap."
    icon = "GoogleGenerativeAI"
    name = "GoogleGenerativeAIModelFactory"

    inputs = [
        *LCModelComponent._base_inputs,
        IntInput(
            name="max_output_tokens", display_name="Max Output Tokens", info="The maximum number of tokens to generate."
        ),
        DropdownInput(
            name="model",
            display_name="Model",
            info="The name of the model to use.",
            options=["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-1.0-pro-vision"],
            value="gemini-1.5-pro",
        ),
        MultilineSecretInput(
            name="google_api_keys",
            display_name="Google API Keys",
            info="The Google API Key to use for the Google Generative AI.",
        ),
        SecretStrInput(
            name="google_auth_postfix",
            display_name="Google Auth Postfix",
            info="The Google Auth Postfix to use for the Google Generative AI.",
        ),
        SecretStrInput(
            name="google_base_url",
            display_name="Google Base Endpoint",
            info="The Google Base Endpoint"
        ),
        StrInput(
            name="google_transport",
            display_name="Google Transport",
            info="The Google Transport",
            value="rest"
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            info="The maximum cumulative probability of tokens to consider when sampling.",
            advanced=True,
        ),
        FloatInput(name="temperature", display_name="Temperature", value=0.1),
        IntInput(
            name="n",
            display_name="N",
            info="Number of chat completions to generate for each prompt. "
            "Note that the API may not return the full n completions if duplicates are generated.",
            advanced=True,
        ),
        IntInput(
            name="top_k",
            display_name="Top K",
            info="Decode using top-k sampling: consider the set of top_k most probable tokens. Must be positive.",
            advanced=True,
        ),
        HandleInput(
            name="output_parser",
            display_name="Output Parser",
            info="The parser to use to parse the output of the model",
            advanced=True,
            input_types=["OutputParser"],
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        model = self.model
        max_output_tokens = self.max_output_tokens
        temperature = self.temperature
        top_k = self.top_k
        top_p = self.top_p
        n = self.n

        factory = GeminiFactory(api_keys=MultilineSecretInput(self.google_api_keys).get_secret_value(), 
                                auth_postfix=SecretStrInput(self.google_auth_postfix).get_secret_value(),
                                base_url=SecretStrInput(self.google_base_url).get_secret_value(),
                                transport=self.google_transport)


        return factory(
            model=model,
            max_output_tokens=max_output_tokens or None,
            temperature=temperature,
            top_k=top_k or None,
            top_p=top_p or None,
            n=n or 1,
        )
