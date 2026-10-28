"""NARVAL-pair pilots: subset rule, preparation and run configs (Modal launcher helpers).

Qwen3-1.7B -> Qwen3-0.6B is the default pair (script/modal/qwen3_pilot.py); PAIRS adds
Qwen3-8B -> Qwen3-1.7B and Llama-3.1-8B -> Llama-3.2-1B (script/modal/narval_pair.py).
Pure functions so the launcher logic is tested on CPU. Every value that is not verified against
NARVAL is listed in the pilot assumptions and recorded in each generated config.
"""
from __future__ import annotations

import copy
from typing import Any

from .artifacts import digest

PAIR = 'qwen3_1p7b_to_0p6b'
TEACHER_BASE_MODEL = 'Qwen/Qwen3-1.7B-Base'
STUDENT_BASE_MODEL = 'Qwen/Qwen3-0.6B-Base'
REWARD_MODEL = 'sfairXC/FsfairX-LLaMA3-RM-v0.1'
SFT_DATASET = 'hkust-nlp/deita-10k-v0'
PREFERENCE_DATASET = 'HuggingFaceH4/ultrafeedback_binarized'
EOS_TOKEN = '<|im_end|>'
MAX_LENGTH = 2048
SUBSET = {'train':2000,'head_val':200,'select':64}
SFT_RECIPE = {'epochs':3,'learning_rate':2e-5,'batch_size':128}
DPO_RECIPE = {'epochs':2,'learning_rate':5e-7,'batch_size':128,'beta':.05}
# Tuned policy settings (not NARVAL's): peak lr 1e-6 with 10% warmup and cosine decay averages
# about 5.5e-7, close to the NARVAL launcher's 5e-7; bf16 autocast and microbatches change speed only.
POLICY_TUNING = {'learning_rate':1e-6,'lr_schedule':'cosine','warmup_ratio':.1,
                 'policy_autocast_dtype':'bfloat16','policy_microbatch_size':8}
PILOT_ASSUMPTIONS = {
    'base_checkpoints':f'{TEACHER_BASE_MODEL} and {STUDENT_BASE_MODEL} (Base vs post-trained is an open NARVAL input)',
    'teacher_pair':'T0 = SFT of Qwen3-1.7B-Base, T+ = DPO of T0; NARVAL derives them from the 8B->1.7B student instead',
    'sft_recipe':'3 epochs, lr 2e-5, effective batch 128 (ADPA/TVKD recipe of the in-repo launcher), max_length 2048',
    'teacher_dpo_recipe':'beta 0.05, lr 5e-7, 2 epochs, effective batch 128, max_length 2048, all UltraFeedback train_prefs',
    'sft_data':('Deita-10K-V0 ShareGPT rows; latest assistant turn whose conversation fits 2048 tokens supervised '
                '(earlier turns as context, no response cut); first answer per prompt kept'),
    'precision':'SFT/DPO and student forwards under bf16 autocast with FP32 weights/optimizer; frozen scoring FP32',
    'policy_tuning':'lr 1e-6 peak, 10% linear warmup, cosine decay, policy microbatch 8; alpha 0.3 unchanged',
    'pilot_subset':f'{SUBSET} prompts per split, ranked by sha256([seed, pilot, group_id])',
    'selection':'FsfairX RM on 64 select prompts every 250 steps (greedy, 1024 new tokens)',
    'eos':'<|im_end|> forced as EOS for every role (Base tokenizers declare <|endoftext|>)'}

# One tokenizer (and chat template) serves every role of a pair; the SFT/DPO checkpoints save it.
PAIRS = {
    PAIR:{'teacher':TEACHER_BASE_MODEL,'student':STUDENT_BASE_MODEL,'tokenizer':STUDENT_BASE_MODEL,
          'eos':EOS_TOKEN,'chat_template_kwargs':{'enable_thinking':False},'gated':False,'generation_batch_size':64},
    'qwen3_8b_to_1p7b':{'teacher':'Qwen/Qwen3-8B-Base','student':'Qwen/Qwen3-1.7B-Base','tokenizer':'Qwen/Qwen3-1.7B-Base',
                        'eos':EOS_TOKEN,'chat_template_kwargs':{'enable_thinking':False},'gated':False,
                        'generation_batch_size':16},
    # Base Llama repos ship no chat template, so the Llama-3.2-1B-Instruct tokenizer (same vocabulary)
    # is shared; its template inserts today's date unless date_string is pinned.
    'llama31_8b_to_llama32_1b':{'teacher':'meta-llama/Llama-3.1-8B','student':'meta-llama/Llama-3.2-1B',
                                'tokenizer':'meta-llama/Llama-3.2-1B-Instruct','eos':'<|eot_id|>',
                                'chat_template_kwargs':{'date_string':'26 Jul 2024'},'gated':True,
                                'generation_batch_size':16},
}


def pilot_assumptions(pair: str = PAIR) -> dict[str,str]:
    """Unverified settings of one pair's pilot (the default pair keeps PILOT_ASSUMPTIONS exactly)."""
    if pair == PAIR:
        return PILOT_ASSUMPTIONS
    spec = PAIRS[pair]
    return {**PILOT_ASSUMPTIONS,
            'base_checkpoints':f"{spec['teacher']} and {spec['student']} (Base vs post-trained is an open NARVAL input)",
            'teacher_pair':f"T0 = SFT of {spec['teacher']}, T+ = DPO of T0 (same recipe as every pair)",
            'eos':(f"{spec['eos']} forced as EOS for every role; tokenizer and chat template of {spec['tokenizer']} "
                   f"with {spec['chat_template_kwargs']} shared by all roles")}


def pilot_subset(rows: list[dict[str,Any]], seed: int = 0, sizes: dict[str,int] | None = None) -> list[dict[str,Any]]:
    """Pre-generation subset: per split, the first n prompt groups ranked by hash (never test)."""
    sizes = sizes or SUBSET
    chosen = []
    for split,size in sizes.items():
        groups = sorted({r.get('group_id',r['id']) for r in rows if r['split'] == split},
                        key=lambda g:digest([seed,'pilot',g]))[:size]
        keep = set(groups)
        chosen += [r for r in rows if r['split'] == split and r.get('group_id',r['id']) in keep]
    return sorted(chosen,key=lambda r:r['id'])


def preparation_config(stage: str, model: str, revision: str, data: str, output: str, pair: str = PAIR,
                       tokenizer: str | None = None, tokenizer_revision: str | None = None) -> dict[str,Any]:
    """Config for train-sft / train-teacher-dpo with the assumed recipe, EOS override and TF32.

    `tokenizer` replaces the model's own tokenizer (Base Llama repos have no chat template);
    DPO starts from the SFT checkpoint, which already saved the shared tokenizer.
    """
    recipe = DPO_RECIPE if stage == 'train-teacher-dpo' else SFT_RECIPE
    cfg = {'model':model,'revision':revision,'data':data,'device':'cuda','seed':0,'output':output,
           **recipe,'eos_token':PAIRS[pair]['eos'],'max_length':MAX_LENGTH,'allow_tf32':True,'autocast_dtype':'bfloat16'}
    if tokenizer is not None:
        cfg.update(tokenizer=tokenizer,tokenizer_revision=tokenizer_revision)
    return cfg


def pilot_config(protocol: dict[str,Any], variant: str, checkpoints: dict[str,dict[str,str]],
                 revisions: dict[str,str], split_manifest_sha256: str, reference_verified: bool,
                 pair: str = PAIR) -> dict[str,Any]:
    """Resolved pilot config from config/crpd/narval/<pair>.json (relative paths only)."""
    spec = PAIRS[pair]
    cfg = copy.deepcopy(protocol)
    root = f'output/narval/{pair}/pilot'
    deviation = ('teacher pair trained directly from Qwen3-1.7B-Base, not taken from the 8B->1.7B protocol' if pair == PAIR
                 else f"teacher pair trained from {spec['teacher']} with the assumed recipe, not NARVAL's checkpoints")
    cfg.update(pilot=True,variant=variant,output=f'{root}/{variant}/seed0',cache_root=f'{root}/cache',
               prompts=f'data/narval/{pair}/ultrafeedback_train_pilot.jsonl',selection_interval=250,
               generation_batch_size=spec['generation_batch_size'],teacher_dpo_reference_verified=reference_verified,
               eos_token=spec['eos'],chat_template_kwargs=copy.deepcopy(spec['chat_template_kwargs']),**POLICY_TUNING,
               models={role:{'path':value['path'],'revision':value['revision']} for role,value in checkpoints.items()},
               pilot_assumptions=pilot_assumptions(pair),pilot_deviations=[deviation])
    cfg['selection']['reward_model_revision'] = revisions['reward_model']
    narval = cfg['narval']
    narval['base_checkpoints'] = {'teacher':f"{spec['teacher']}@{revisions['teacher_base_model']}",
                                  'student':f"{spec['student']}@{revisions['student_base_model']}"}
    narval['sft'].update(dataset_revision=revisions['sft_dataset'],max_length=MAX_LENGTH,**SFT_RECIPE)
    narval['teacher_dpo'].update(dataset_revision=revisions['preference_dataset'],max_length=MAX_LENGTH,
                                 **{k:DPO_RECIPE[k] for k in ('beta','learning_rate','epochs')})
    narval['preference_data'].update(revision=revisions['preference_dataset'],split_manifest_sha256=split_manifest_sha256)
    narval['provenance'] = {}
    return cfg
