# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class forFireRed.
"""

from typing import Union

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class FireRedProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "audio_kwargs": {},
    }


class FireRedProcessor(ProcessorMixin):
    r"""
    Constructs aFireRed processor which wraps aFireRed feature extractor and aFireRed tokenizer into a single processor.

    [`Qwen2AudioProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2AudioProcessor.__call__`] and [`~Qwen2AudioProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the default chat template
                is used.
        audio_token (`str`, *optional*, defaults to `"<|AUDIO|>"`):
            The token to use for audio tokens.
        audio_bos_token (`str`, *optional*, defaults to `"<|audio_bos|>"`):
            The token to use for audio bos tokens.
        audio_eos_token (`str`, *optional*, defaults to `"<|audio_eos|>"`):
            The token to use for audio eos tokens.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer=None,
        chat_template=None,
        audio_token="<speech>",
    ):
        if chat_template is None:
            chat_template = self.default_chat_template
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        super().__init__(tokenizer, chat_template=chat_template)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["feature_attention_mask"]))

    @property
    # NOTE: we don't have default templates anymore, and the below is kept only because the hub config is not yet updated!
    def default_chat_template(self):
        """
        This default vicuna template formats inputs in the form of a chat history. For each message in the chat history:
        * the template will output the role of the speaker followed by the content of the message.
        * content is a list of strings and audios.
        * If the content element is an audio, the template will output a sequence of <|AUDIO|> tokens

        Example:

        ```python
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
                {"type": "text", "text": "What's that sound?"},
            ]},
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"},
                {"type": "text", "text": "How about this one?"},
            ]},
        ]

        result = template.render(messages=messages, add_generation_prompt=True)
        ```
        """
        # fmt: off
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content or message['type'] == 'audio' or content['type'] == 'audio' %}"
                            "{% set audio_count.value = audio_count.value + 1 %}"
                            "Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
        )
        # fmt: on


__all__ = ["FireRedProcessor"]
