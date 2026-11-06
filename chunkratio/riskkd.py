"""CRPD on the NARVAL (riskKD) authors' published checkpoints for the three Table 1 pairs (DECISIONS 93).

Every role is a pinned vukien2301 repo listed in riskKD_hf_checkpoints.xlsx; nothing is trained before CRPD.
Within a pair the three checkpoints share one tokenizer, vocabulary and chat template (checked through the Hub on
6 November 2026), and the end-of-turn token of that template is the EOS of every role. `NARVAL` names the
authors' student checkpoint used as the comparison row in the benchmarks. Used by script/server/run_pair.py; the
Modal launchers keep their own settings.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from .pilot import HDA_ASSUMPTIONS, HDA_CHECKPOINTS, HDA_PAIR, PAIRS, pilot_config

U = 'vukien2301/'
RISKKD = {
    'qwen3_1p7b_to_0p6b':{
        'protocol':'qwen3_1p7b_to_0p6b','run_pair':HDA_PAIR,'eos':'<|im_end|>',
        'chat_template_kwargs':{'enable_thinking':False},
        'checkpoints':HDA_CHECKPOINTS,
        'narval':{'repo':U+'qwen3-0.6b-riskkd','revision':'e3936e47a86bf7d088bc6a70ea7bcc327e93e0a6'},
        'reference_alternatives':['Qwen/Qwen3-1.7B-Base','Qwen/Qwen3-1.7B']},
    'qwen3_8b_to_1p7b':{
        'protocol':'qwen3_8b_to_1p7b','run_pair':'qwen3_8b_to_1p7b_riskkd','eos':'<|im_end|>',
        'chat_template_kwargs':{'enable_thinking':False},
        'checkpoints':{
            'teacher_base':{'repo':U+'qwen3-8b-deita-sft-teacher','revision':'0992ec63a179eef82d1fcbc0188d6cabf44f76d7'},
            'teacher_plus':{'repo':U+'qwen3-8b-ultrafeedback-dpo-teacher','revision':'e7c6d8deea8192057d82768e6d282d82dac7c140'},
            'student_ref':{'repo':U+'qwen3-1.7b-deita-sft-student','revision':'68735fe3c227a7c62c0f34fe6664de8b3714b039'}},
        'narval':{'repo':U+'qwen3-1.7b-tailriskKD','revision':'b6dc675f84e527fbba5bac915d998f234019ca2b'},
        'reference_alternatives':['Qwen/Qwen3-8B-Base','Qwen/Qwen3-8B']},
    'llama31_8b_to_llama32_1b':{
        # The SFT teacher is epoch 1 of a 3-epoch run; the DPO teacher was trained from it ("the teacher used for
        # the Llama students", per the spreadsheet). The shared template ends every turn with <|eot_id|>.
        'protocol':'llama31_8b_to_llama32_1b','run_pair':'llama31_8b_to_llama32_1b_riskkd','eos':'<|eot_id|>',
        # The authors' template inserts no date; date_string is pinned anyway, as the protocol requires of Llama 3
        # templates (unused here, so the prompt tokens are those of the template).
        'chat_template_kwargs':{'date_string':'26 Jul 2024'},
        'checkpoints':{
            'teacher_base':{'repo':U+'llama-3.1-8b-deita-sft-teacher-epoch1','revision':'d37182e703564c0b09436cb158962ebf43076065'},
            'teacher_plus':{'repo':U+'llama-3.1-8b-ultrafeedback-dpo-from-epoch1','revision':'9e4fb7411c432319561fd7caa482ccf06642f2e9'},
            'student_ref':{'repo':U+'llama-3.2-1b-deita-sft-student','revision':'5900e0a4fa809ab32a534ad8a5003c9a2bb28e32'}},
        # Its AlpacaEval result equals the NARVAL row of paper Table 3 (spreadsheet note).
        'narval':{'repo':U+'llama3.2-1b-tailriskKD','revision':'9bd3a37d8f2312fbf122c5718a3dde514fc99aef'},
        # Both are gated: the weight check needs an HF token whose account accepted the Llama 3.1 licence.
        'reference_alternatives':['meta-llama/Llama-3.1-8B','meta-llama/Llama-3.1-8B-Instruct']},
}
for _name,_spec in RISKKD.items():
    if _spec['run_pair'] not in PAIRS:
        _ckpt = _spec['checkpoints']
        PAIRS[_spec['run_pair']] = {'teacher':_ckpt['teacher_base']['repo'],'student':_ckpt['student_ref']['repo'],
                                    'tokenizer':_ckpt['teacher_base']['repo'],'eos':_spec['eos'],
                                    'chat_template_kwargs':copy.deepcopy(_spec['chat_template_kwargs']),'gated':False,
                                    'generation_batch_size':32}

# Prompt subsets. x10 and x30 are the Modal pilot sizes (same 20,480 UltraFeedback rows, so their select and test
# prompts agree); full uses every eligible train prompt. Selection always uses the same 64 select prompts.
SCALES = {
    'x10':{'uf_train_rows':20480,'subset':{'train':640,'head_val':160,'select':64}},
    'x30':{'uf_train_rows':20480,'subset':{'train':2000,'head_val':200,'select':64}},
    'full':{'uf_train_rows':None,'subset':{'train':None,'head_val':None,'select':64}},
}
UF_TEST_ROWS = 64
# The measured turbo settings of the Modal runs (results unchanged up to FP32 rounding). B200s have more memory;
# the token budgets stay at the A100-80GB values unless the run overrides them (run_pair.py --policy-tokens etc.).
SPEED = {'gpu_sampling':True,'generation_tokens_per_batch':163840,'generation_max_rows':384,
         'policy_tokens_per_microbatch':8192,'batch_candidate_scoring':True,'score_tokens_per_batch':16384,
         'selection_generation_batch_size':32,'shared_prefix_scoring':True,'score_max_states':48,
         'max_prompt_tokens':2048,'states_per_prompt':2,'selection_baseline':True}


def assumptions(pair: str) -> dict[str,str]:
    """Unverified settings recorded in every config of the pair."""
    spec = RISKKD[pair]
    ckpt = spec['checkpoints']
    return {**{k:v for k,v in HDA_ASSUMPTIONS.items() if k not in ('teacher_pair','sft_data','eos')},
            'teacher_pair':(f"T0 = {ckpt['teacher_base']['repo']}, T+ = {ckpt['teacher_plus']['repo']} (riskKD authors' "
                            "checkpoints, riskKD_hf_checkpoints.xlsx). That T0 was T+'s DPO reference is not documented"),
            'sft_data':f"rho = {ckpt['student_ref']['repo']} (riskKD student init), not trained here",
            'eos':f"{spec['eos']} for every role (end of each turn in the shared template)"}


def steps_for(pairs: int, batch_size: int, passes: float) -> int:
    """Optimizer steps covering `passes` passes over the training pairs (rounded up)."""
    return max(1,math.ceil(passes*pairs/batch_size))


def server_config(pair: str, variant: str, revisions: dict[str,str], split_hash: str, prompts: str, output: str,
                  cache_root: str, *, alpha: float, gamma: float = 1., pair_sources: str = 'all', steps: int = 150,
                  batch_size: int = 128, selection_interval: int | None = None, protocol: dict[str,Any],
                  speed: dict[str,Any] | None = None, collection_splits: list[str] | None = None,
                  evidence: dict[str,Any]) -> dict[str,Any]:
    """Resolved run config: NARVAL protocol file + pinned riskKD trio + speed settings + this run's choices.

    `evidence` is the weight-space check of T0 as T+'s DPO reference (run_pair.py verify); the run records the
    reference as an assumption with that evidence, never as verified."""
    spec = RISKKD[pair]
    checkpoints = {role:{'path':source['repo'],'revision':source['revision']} for role,source in spec['checkpoints'].items()}
    assumption = {'status':'assumed','claim':"T0 is T+'s actual DPO reference",
                  'evidence':{'relative_l2_distance':evidence['relative_l2_distance'],'rule':evidence['rule']},
                  'not_verified_because':'the riskKD repos publish no adapter, reference model name or training config'}
    cfg = pilot_config(protocol,variant,checkpoints,revisions,split_hash,False,spec['run_pair'],assumption)
    cfg.update(SPEED,**(speed or {}))
    cfg.update(alpha=alpha,steps=steps,batch_size=batch_size,selection_interval=selection_interval or steps,
               prompts=prompts,output=output,cache_root=cache_root,pilot_assumptions=assumptions(pair))
    if gamma != 1:
        cfg['teacher_gap_scale'] = gamma
    if pair_sources != 'all':
        cfg['pair_sources'] = pair_sources
    if collection_splits is not None:
        cfg['collection_splits'] = collection_splits
    # A finished student holds the same weights as its HF export.
    cfg['save_student_state_dict'] = False
    return cfg
