# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Registration function for explicit use
def register_firered():
    """Register FunAudioChat classes with AutoConfig and AutoModel."""
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
    from .configuration_firered import FireredAudioConfig, FireredAudioEncoderConfig
    from .modeling_firered import FireRedForConditionalGeneration
    from .processing_firered import FireRedProcessor
    
    # Register configurations
    AutoConfig.register("firered_audio", FireredAudioConfig)
    AutoConfig.register("firered_audio_encoder", FireredAudioEncoderConfig)
    
    # Register processor
    AutoProcessor.register(FireredAudioConfig, FireRedProcessor)
    
    # Register model
    AutoModelForSeq2SeqLM.register(FireredAudioConfig, FireRedForConditionalGeneration)