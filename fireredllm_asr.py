from functools import partial
from typing import Any, Dict, Optional, Literal, List, Union
import numpy as np
import os
import kaldiio
import kaldi_native_fbank as knf
import math
from types import MethodType
import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
import sys
import torch
from packaging import version
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence


from swift.llm import (Model, ModelGroup, ModelMeta, MultiModelKeys, Template, TemplateMeta, get_model_tokenizer, get_template, register_model,
                       register_model_arch, register_template, get_model_tokenizer_with_flash_attn)
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.llm.template.vision_utils import load_audio, load_batch
from swift.utils import get_env_args, get_logger

logger = get_logger()


logger.info("Registering FireRed with transformers AutoClasses...")
from fireredasr_model.register import register_firered
register_firered()
logger.info("✓ FireRed registered with AutoProcessor AutoConfig and AutoModelForSeq2SeqLM")


def get_model_tokenizer_firered(model_dir, *args, **kwargs):
    from fireredasr_model import FireRedForConditionalGeneration, FireRedProcessor, FireredAudioConfig
    print('Run firered...')
    kwargs['automodel_class'] = kwargs['automodel_class'] or FireRedForConditionalGeneration

    processor = FireRedProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    kwargs['model_config'] = FireredAudioConfig.from_pretrained(model_dir, trust_remote_code=True)
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    if model:
        model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
    return model, processor


'''
首先注册模型
'''
# 注册模型结构
register_model_arch(
    MultiModelKeys(
        'fireredllm_asr',
        language_model='language_model',
        aligner='multi_modal_projector',
        vision_tower='audio_tower',
    ))



register_model(
    ModelMeta(
        model_type='fireredllm_asr',
        model_groups=[ModelGroup([Model('FireRedASR')])],
        template='fireredllm_asr',
        get_function=get_model_tokenizer_firered,
        is_multimodal=True,                                             # 这个地方是问题的关键哦
        model_arch='fireredllm_asr',
        architectures=['FireRedForConditionalGeneration'],
        requires=['transformers>=4.45,<4.49', 'librosa'],
        tags=['audio'],
    ))


class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = \
            self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variences = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            varience = (stats[1, d] / count) - mean*mean
            if varience < floor:
                varience = floor
            istd = 1.0 / math.sqrt(varience)
            inverse_std_variences.append(istd)
        return dim, np.array(means), np.array(inverse_std_variences)

class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10,
                 dither=1.0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav, is_train=False):
        
        if type(wav) is str:
            sample_rate, wav_np = kaldiio.load_mat(wav)
        elif type(wav) in [tuple, list] and len(wav) == 2:
            sample_rate, wav_np = wav
        assert len(wav_np.shape) == 1

        dither = self.dither if is_train else 0.0
        self.opts.frame_opts.dither = dither
        fbank = knf.OnlineFbank(self.opts)

        fbank.accept_waveform(sample_rate, wav_np.tolist())
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if len(feat) == 0:
            print("Check data, len(feat) == 0", wav, flush=True)
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack(feat)
        return feat


'''
然后注册模板
'''
class Qwen2AudioTemplate(Template):
    
    def init_env_args(self) -> None:
        super().init_env_args()
        
        self.sampling_rate = get_env_args('sampling_rate', int, 16000)
        self.cmvn = CMVN('cmvn.ark')
        self.fbank = KaldifeatFbank(num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0)
        
    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        return ['<speech>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)       # 处理纯文本部分，具体请参考自定义模型文档 
        
        if inputs.audios: 
            input_features = []
            audios = load_batch(inputs.audios, load_func=partial(load_audio, sampling_rate=self.sampling_rate))
            
            for audio in audios:
                audio = audio * (1 << 15)
                fbank = self.fbank((self.sampling_rate, audio))
                if self.cmvn is not None:
                    fbank = self.cmvn(fbank)
                fbank = torch.from_numpy(fbank).float()
                input_features.append(fbank)
                
            input_features = self.pad_feat(input_features, 0.0)   
            encoded.update({'input_features': input_features})
            
        encoded['input_ids_length'] = encoded['input_ids']
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        audio_token_id = self._tokenize('<speech>')
        idx_list = findall(input_ids, audio_token_id) 
        audio_lengths = input_features.shape[1] // 8
        if idx_list:
            def _get_new_audio_tokens(i):
                return audio_token_id * audio_lengths
            
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_audio_tokens)
            
        encoded['input_ids'] = input_ids
        
        return encoded
    
    def pad_feat(self, xs, pad_value):
        n_batch = len(xs)
        max_len = max([xs[i].size(0) for i in range(n_batch)])
        pad = torch.ones(n_batch, max_len, *xs[0].size()[1:]).to(xs[0].device).to(xs[0].dtype).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]
        return pad

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        
        if input_features:
            feats_lengths = torch.tensor([input_feature.size(1) for input_feature in input_features], dtype=torch.int32)
            feature_attention_mask = ~self.make_pad_mask(feats_lengths)

            input_features = [input_feature.squeeze(0) for input_feature in input_features]
            res['input_features'] = pad_sequence(input_features, batch_first=True) 
            res['feats_lengths'] = feats_lengths
            res['feature_attention_mask'] = feature_attention_mask
        return res
    
    def make_pad_mask(self, lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
        """Make mask tensor containing indices of padded part.

        See description of make_non_pad_mask.

        Args:
            lengths (torch.Tensor): Batch of lengths (B,).
        Returns:
            torch.Tensor: Mask tensor containing indices of padded part.

        Examples:
            >>> lengths = [5, 3, 2]
            >>> make_pad_mask(lengths)
            masks = [[0, 0, 0, 0 ,0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 1, 1]]
        """
        batch_size = lengths.size(0)
        max_len = max_len if max_len > 0 else lengths.max().item()
        seq_range = torch.arange(0,
                                max_len,
                                dtype=torch.int64,
                                device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_length_expand = lengths.unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand
        return mask
    

register_template(
    TemplateMeta(
        'fireredllm_asr',
        prefix=[],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|im_end|>'],                              # 最后加
        system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        default_system='',
        stop_words=['<|endoftext|>'],
        agent_template='hermes',
        template_cls=Qwen2AudioTemplate))


if __name__ == '__main__':
    # 测试与debug
    model, processor = get_model_tokenizer('Qwen/Qwen2-Audio-7B-Instruct', model_type='fireredllm_asr')
    template = get_template('fireredllm_asr', processor)
    
    data = {
        'messages': [
            {'role': 'user', 'content': '描述视频图片<audio>内容。'},
            {'role': 'assistant', 'content': '一个小孩和一只猫咪。'},
        ],
        'audios': [''],
    }
    
    template.set_mode('train')
    encoded = template.encode(data)
    print('input_ids: ' + template.safe_decode(encoded['input_ids']))
    print('labels: ' + template.safe_decode(encoded['labels']))
    print('keys: ' + str(encoded.keys()))
