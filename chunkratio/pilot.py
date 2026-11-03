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
# about 5.5e-7, close to the NARVAL launcher's 5e-7; microbatches change speed only, and bf16 autocast
# also rounds the policy forwards (student and rho alike, so the margin still starts at zero).
POLICY_TUNING = {'learning_rate':1e-6,'lr_schedule':'cosine','warmup_ratio':.1,
                 'policy_autocast_dtype':'bfloat16','policy_microbatch_size':8}
PILOT_ASSUMPTIONS = {
    'base_checkpoints':f'{TEACHER_BASE_MODEL} and {STUDENT_BASE_MODEL} (Base vs post-trained is an open NARVAL input)',
    'teacher_pair':'T0 = SFT of Qwen3-1.7B-Base, T+ = DPO of T0; NARVAL derives them from the 8B->1.7B student instead',
    'sft_recipe':'3 epochs, lr 2e-5, effective batch 128 (ADPA/TVKD recipe of the in-repo launcher), max_length 2048',
    'teacher_dpo_recipe':'beta 0.05, lr 5e-7, 2 epochs, effective batch 128, max_length 2048, all UltraFeedback train_prefs',
    'sft_data':('Deita-10K-V0 ShareGPT rows; latest assistant turn whose conversation fits 2048 tokens supervised '
                '(earlier turns as context, no response cut); first answer per prompt kept'),
    'precision':('SFT/DPO and policy forwards (student and rho) under bf16 autocast with FP32 weights/optimizer; '
                 'collection and target scoring FP32'),
    'policy_tuning':'lr 1e-6 peak, 10% linear warmup, cosine decay, policy microbatch 8; alpha 0.3 unchanged',
    'pilot_subset':f'{SUBSET} prompts per split, ranked by sha256([seed, pilot, group_id])',
    'selection':'FsfairX RM on 64 select prompts every 250 steps (greedy, 1024 new tokens)',
    'eos':'<|im_end|> forced as EOS for every role (Base tokenizers declare <|endoftext|>)'}

# One tokenizer (and chat template) serves every role of a pair; the SFT/DPO checkpoints save it.
PAIRS = {
    PAIR:{'teacher':TEACHER_BASE_MODEL,'student':STUDENT_BASE_MODEL,'tokenizer':STUDENT_BASE_MODEL,
          # 32 rows: FP32 KV caches of ~1.5k-token Qwen3 prefixes take ~229 KB/token (64 states x 2 rows OOMed).
          'eos':EOS_TOKEN,'chat_template_kwargs':{'enable_thinking':False},'gated':False,'generation_batch_size':32},
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

# Published Qwen3-1.7B teacher pair (pinned revisions): T+ is a LoRA DPO of T0, and the LoRA is
# published too, so chunkratio.hub_teachers can check T+ = T0 + LoRA from the weights.
HUB_TEACHERS = {
    'teacher_base':{'repo':'ayushshah/Qwen3-1.7B-UltraChat-SFT','revision':'698d09a459656c6909e2b79c6827b66c7a10d70b'},
    'teacher_plus':{'repo':'ayushshah/Qwen3-1.7B-Chat','revision':'2e46c30d9fba4ed1aaa8f21003e0de58fa7cf944'},
    'dpo_lora':{'repo':'ayushshah/Qwen3-1.7B-Chat-LoRA','revision':'c1a6e68d4e8adb95ef8c535d705985da6c7b6f15'},
}
HUB_PAIR = PAIR+'_hub'
# rho is our own SFT of Qwen3-0.6B-Base on UltraChat with the teachers' plain ChatML template: the
# official Qwen3-0.6B uses a thinking template whose prompt tokens differ from the teachers'.
PAIRS[HUB_PAIR] = {'teacher':HUB_TEACHERS['teacher_base']['repo'],'student':STUDENT_BASE_MODEL,
                   'tokenizer':HUB_TEACHERS['teacher_base']['repo'],'eos':EOS_TOKEN,
                   'chat_template_kwargs':{'enable_thinking':False},'gated':False,'generation_batch_size':32}
HUB_ASSUMPTIONS = {
    **{k:v for k,v in PILOT_ASSUMPTIONS.items() if k not in ('teacher_pair','teacher_dpo_recipe','sft_data','sft_recipe')},
    'base_checkpoints':f'{STUDENT_BASE_MODEL} for rho; teachers are published checkpoints, not trained here',
    'teacher_pair':('T0 = ayushshah/Qwen3-1.7B-UltraChat-SFT (UltraChat-200k SFT of Qwen3-1.7B-Base); '
                    'T+ = ayushshah/Qwen3-1.7B-Chat (LoRA r32/a64 DPO of T0, lr 5e-6, 1 epoch, argilla UltraFeedback-'
                    'cleaned 15k + Capybara + Orca pairs); T+ = T0 + LoRA checked from the weights before use'),
    'sft_data':('rho: first 10,000 HuggingFaceH4/ultrachat_200k train_sft conversations, latest assistant turn that '
                'fits 2048 tokens supervised, teachers\' plain ChatML template'),
    'sft_recipe':'rho: 3 epochs, lr 2e-5, effective batch 128, full fine-tuning (not the teachers\' LoRA SFT recipe)',
    'eos':'<|im_end|> for every role; the teachers\' tokenizer and plain ChatML template are shared by all roles',
    'pilot_subset':"{'train': 1000, 'head_val': 100, 'select': 64} prompts, ranked by sha256([seed, pilot, group_id])",
    'states_per_prompt':'2 (pre-registered 4 halved for cost; recorded before collection)',
    'policy_steps':'300 steps x 64 pairs; RM selection every 150 steps; variants crpd_main and crpd_raw',
    'contamination':('T+ saw argilla UltraFeedback-cleaned prompts, which share the UltraFeedback prompt pool with '
                     'H4 test_prefs; held-out prompts seen by T+ must be excluded before reporting'),
}


# NARVAL-matching published trio (pinned revisions, 3 November 2026): Deita SFT of Qwen3-1.7B-Base (T0),
# its UltraFeedback DPO (T+) and a Deita SFT of Qwen3-0.6B-Base (rho). All three share one tokenizer and
# a plain ChatML template whose assistant turns end with <|im_end|>; the tokenizers declare <|endoftext|>,
# so <|im_end|> is set as EOS when loading. On Winogrande and TruthfulQA-MC2 the 1.7B pair reproduces the
# NARVAL Table 1 teacher rows within 0.6 points (script/modal/benchmark_eval.py).
HDA_CHECKPOINTS = {
    'teacher_base':{'repo':'hoangducanh1865/qwen3-1.7b-deita-sft-student','revision':'52f7c14ea4893128252ada7a365c957813820070'},
    'teacher_plus':{'repo':'hoangducanh1865/qwen3-1.7b-ultrafeedback-dpo-teacher','revision':'97a6de251a5e3f78494c6892fb4fbdf9b53c9f04'},
    'student_ref':{'repo':'hoangducanh1865/qwen3-0.6b-deita-sft-student','revision':'fc0f225cb63e5db6e84a7e5705c997891ec85d65'},
}
HDA_PAIR = PAIR+'_hda'
PAIRS[HDA_PAIR] = {'teacher':HDA_CHECKPOINTS['teacher_base']['repo'],'student':HDA_CHECKPOINTS['student_ref']['repo'],
                   'tokenizer':HDA_CHECKPOINTS['teacher_base']['repo'],'eos':EOS_TOKEN,
                   'chat_template_kwargs':{'enable_thinking':False},'gated':False,'generation_batch_size':32}
HDA_ASSUMPTIONS = {
    **{k:v for k,v in PILOT_ASSUMPTIONS.items() if k not in ('teacher_pair','teacher_dpo_recipe','sft_data','sft_recipe',
                                                             'base_checkpoints')},
    'base_checkpoints':'all three roles are published checkpoints; nothing is trained before CRPD',
    'teacher_pair':('T0 = hoangducanh1865/qwen3-1.7b-deita-sft-student (Deita-10k SFT of Qwen3-1.7B-Base, 1 epoch, '
                    'lr 3e-5, per its model card); T+ = hoangducanh1865/qwen3-1.7b-ultrafeedback-dpo-teacher (1 epoch '
                    'DPO on 57,673 UltraFeedback pairs per its trainer_state; no model card). That T0 was T+\'s DPO '
                    'reference is not documented: the launcher only checks that T0 is the nearest candidate to T+ '
                    'in weight space'),
    'sft_data':'rho: hoangducanh1865/qwen3-0.6b-deita-sft-student (Deita SFT of Qwen3-0.6B-Base), not trained here',
    'sft_recipe':'not applicable (published rho)',
    'eos':'<|im_end|> for every role (end of each assistant turn in the shared template; tokenizers declare <|endoftext|>)',
}


def pilot_assumptions(pair: str = PAIR) -> dict[str,str]:
    """Unverified settings of one pair's pilot (the default pair keeps PILOT_ASSUMPTIONS exactly)."""
    if pair == PAIR:
        return PILOT_ASSUMPTIONS
    if pair == HUB_PAIR:
        return HUB_ASSUMPTIONS
    if pair == HDA_PAIR:
        return HDA_ASSUMPTIONS
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
                 pair: str = PAIR, reference_assumption: dict[str,Any] | None = None) -> dict[str,Any]:
    """Resolved pilot config from config/crpd/narval/<pair>.json (relative paths only)."""
    spec = PAIRS[pair]
    cfg = copy.deepcopy(protocol)
    root = f'output/narval/{pair}/pilot'
    deviation = ('teacher pair trained directly from Qwen3-1.7B-Base, not taken from the 8B->1.7B protocol' if pair == PAIR
                 else 'published teacher pair (T+ = T0 + LoRA, checked from weights), not NARVAL\'s checkpoints' if pair == HUB_PAIR
                 else ('published T0/T+/rho trio matching NARVAL Table 1 on two benchmarks; T+\'s DPO reference is '
                       'checked only as the nearest candidate in weight space') if pair == HDA_PAIR
                 else f"teacher pair trained from {spec['teacher']} with the assumed recipe, not NARVAL's checkpoints")
    cfg.update(pilot=True,variant=variant,output=f'{root}/{variant}/seed0',cache_root=f'{root}/cache',
               prompts=f'data/narval/{pair}/ultrafeedback_train_pilot.jsonl',selection_interval=250,
               generation_batch_size=spec['generation_batch_size'],teacher_dpo_reference_verified=reference_verified,
               eos_token=spec['eos'],chat_template_kwargs=copy.deepcopy(spec['chat_template_kwargs']),**POLICY_TUNING,
               models={role:{'path':value['path'],'revision':value['revision']} for role,value in checkpoints.items()},
               pilot_assumptions=pilot_assumptions(pair),pilot_deviations=[deviation])
    if reference_assumption is not None:
        # Unverified T0-as-reference: recorded with its evidence so every result of this run carries it.
        cfg['teacher_dpo_reference_assumption'] = copy.deepcopy(reference_assumption)
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
