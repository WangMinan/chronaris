"""SensorLLM with a disclosed, local DeepSeek Llama backbone substitution."""
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn

from chronaris.modeling.training.candidate_checkpoint import atomic_save_candidate


class SensorLLMWindowEncoder(nn.Module):
    """Reuse official multichannel token insertion; train only its alignment MLP."""
    def __init__(self, assets, channel_count):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        source = str(Path(assets['sensorllm']['path']).resolve())
        sys.path.insert(0, source)
        from sensorllm.model import SensorLLMStage2LlamaForSequenceClassification
        from sensorllm.model.stage2_sensorllm import SensorLLMStage2Config, SensorLLMStage2LlamaModel
        from sensorllm.model.chronos_model import ChronosPipeline
        # Transformers 4.57's generic classification parent also resolves the base model.
        AutoModel.register(SensorLLMStage2Config, SensorLLMStage2LlamaModel, exist_ok=True)
        self.assets = assets
        self.tokenizer = AutoTokenizer.from_pretrained(assets['deepseek_llama']['path'], local_files_only=True)
        model = SensorLLMStage2LlamaForSequenceClassification.from_pretrained(
            assets['deepseek_llama']['path'], num_labels=2, torch_dtype=torch.bfloat16,
            device_map='cuda', local_files_only=True, attn_implementation='sdpa')
        self.core = model.get_model()
        self.feature_dim = model.config.hidden_size
        dataset = dict(channel_num=channel_count)
        for c in range(channel_count):
            dataset[f'default_channel_{c}_start_token'] = f'<channel_{c}_start>'
            dataset[f'default_channel_{c}_end_token'] = f'<channel_{c}_end>'
        self.core.ts_backbone_config['chronaris'] = dataset
        model.initialize_tokenizer_ts_backbone_config(self.tokenizer, 'cuda', fix_llm=False, dataset='chronaris')
        self.core.load_start_end_tokens('chronaris')
        # ponytail: freeze mean boundary embeddings for this 24 GB diagnostic;
        # add trainable boundary rows after a separate memory-budget check.
        pipeline = ChronosPipeline.from_pretrained(assets['chronos_t5_large']['path'], device_map='cuda',
            tc='StanNormalizeUniformBins', torch_dtype=torch.bfloat16, local_files_only=True)
        self.ts_tokenizer = pipeline.tokenizer
        self.core.pt_encoder_backbone = pipeline.model
        self.core.fix_ts_encoder = True
        self.core.config.use_cache = False
        self.core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.configure_training()

    def configure_training(self):
        self.requires_grad_(False)
        self.core.ts_proj.requires_grad_(True)

    def checkpoint_state(self):
        return {k: v.detach().cpu() for k, v in self.core.ts_proj.state_dict().items()}

    def load_checkpoint_state(self, state):
        self.core.ts_proj.load_state_dict(state)

    def _input(self, values, mask, names, *, selected_channel=None, answer=None):
        indices = list(range(values.shape[-1])) if selected_channel is None else [selected_channel]
        x = values[:, indices].T.masked_fill(~mask[:, indices].T, torch.nan)
        ts_ids, ts_mask, _ = self.ts_tokenizer.context_input_transform(x)
        parts = ['The following channels contain observations within one completed historical window.']
        for c in indices:
            available = int(mask[:, c].sum())
            parts.append(f'{names[c]}, available grid points {available} of {len(values)}: '
                f'<channel_{c}_start>'+('<ts>'*ts_ids.shape[-1])+f'<channel_{c}_end>')
        parts.append('Describe whether the observed values rise or fall. Answer:' if selected_channel is not None
            else 'Use these observed channels to represent the completed window.')
        prefix = ' '.join(parts)
        text = prefix if answer is None else prefix+' '+answer
        ids = self.tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
        if ids.shape[1] > 4096:
            raise ValueError('SensorLLM diagnostic exceeds its fixed 4096-token resource bound')
        self.last_token_count = ids.shape[1]
        mapping = self.core.start_end_tokens
        if selected_channel is not None:
            start = self.tokenizer.convert_tokens_to_ids(f'<channel_{selected_channel}_start>')
            self.core.start_end_tokens = {start: mapping[start]}
        try:
            hidden = self.core(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                mts_token_ids=ts_ids[None].cuda(), mts_attention_mask=ts_mask[None].cuda()).last_hidden_state
        finally:
            self.core.start_end_tokens = mapping
        return hidden, ids, len(self.tokenizer(prefix, add_special_tokens=False).input_ids)

    def forward(self, values, mask, prompts=None):
        hidden, _, _ = self._input(values[0], mask[0], self.channel_names)
        if len(values) != 1:
            raise ValueError('SensorLLM diagnostic uses single-window batches')
        return hidden[:, -1, :].float() * mask.any().to('cuda')

    def align_history(self, *, values, mask, names, train_ids, root, binding, progress=None):
        """Two single-channel trend QA updates, derived solely from train history."""
        from safetensors import safe_open
        self.channel_names = names
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root/'alignment.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, weights_only=True)
            if saved['binding'] != binding:
                raise ValueError('SensorLLM alignment source changed')
            self.load_checkpoint_state(saved['projector'])
            return saved['receipt']
        index = json.loads((Path(self.assets['deepseek_llama']['path'])/'model.safetensors.index.json').read_text())
        with safe_open(str(Path(self.assets['deepseek_llama']['path'])/index['weight_map']['lm_head.weight']), framework='pt') as file:
            lm_weight = file.get_tensor('lm_head.weight').cuda()
        optimizer = torch.optim.AdamW(self.core.ts_proj.parameters(), lr=2e-3)
        self.train()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        updates = []
        for row in range(2):
            channel = next((c for c in range(values.shape[-1]) if mask[row, :, c].sum() >= 2), None)
            if channel is None:
                raise ValueError('alignment sample lacks observed channel support')
            observed = values[row, :, channel][mask[row, :, channel]]
            answer = 'rising' if observed[-1] > observed[0] else 'falling' if observed[-1] < observed[0] else 'unchanged'
            optimizer.zero_grad(set_to_none=True)
            hidden, ids, boundary = self._input(values[row], mask[row], names, selected_channel=channel, answer=answer)
            # Same causal language loss as the official first stage, only answer tokens scored.
            logits = nn.functional.linear(hidden[:, boundary-1:-1], lm_weight).float()
            loss = nn.functional.cross_entropy(logits.flatten(0,1), ids[:, boundary:].flatten())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.core.ts_proj.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if progress is not None:
                progress['optimizer_updates'] += 1
                progress['alignment_updates'] = row+1
            updates.append(dict(sample_id=train_ids[row], channel=names[channel], answer=answer,
                loss=float(loss.detach()), gradient_norm=float(norm), token_count=self.last_token_count))
        torch.cuda.synchronize()
        receipt = dict(updates=updates, seconds=time.perf_counter()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), supervision='train_history_derived_trend_text',
            business_or_expert_labels=False, special_token_embeddings='frozen_official_mean_initialization')
        atomic_save_candidate(checkpoint, dict(binding=binding, projector=self.checkpoint_state(), receipt=receipt))
        del lm_weight, optimizer
        torch.cuda.empty_cache()
        return receipt
