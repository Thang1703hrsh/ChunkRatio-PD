"""Modal: preference-distillation diagnostics of a finished CRPD run (no training; measurement only).

A. Pair targets by type (rho+rho, rho+teacher, teacher+teacher) and prefix depth: target weight share, how
   often the teacher side is preferred, and the trained student's agreement with the target direction, with
   the student-rho log-ratio of the preferred and the dispreferred side (bf16 autocast as in training).
B. Teacher preference on independent pairs: UltraFeedback train_prefs rows outside the run's prompts, scored
   by T+ and T0 in FP32. Accuracy of delta_T = log T+ - log T0 (summed and per token) in ranking chosen over
   rejected, next to a longer-response baseline and to T+ alone.
C. Free generation on the held-out test prompts (never used for selection): RM reward of rho and the student
   with a paired bootstrap, response length, EOS rate, 4-gram repetition, and the student-rho per-token
   log-ratio on each model's own greedy responses.

  modal deploy script/modal/pd_diagnostics.py && python script/modal/pd_diagnostics.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

APP_NAME = 'crpd-pd-diagnostics'
# CRPD_DIAG_TAG=noend reads crpd_raw-alpha0p03-noend; the default is the first turbo run.
TAG = os.environ.get('CRPD_DIAG_TAG','turbo')
# CRPD_DIAG_RUN names any finished run directory under the pilot root (for example
# crpd_raw-alpha0p03-gamma0p3-gamma); it overrides the tag-derived name and names the output after it.
RUN_NAME = os.environ.get('CRPD_DIAG_RUN','')
NAME = RUN_NAME or f'crpd_raw-alpha0p03-{TAG}'
RUN = f'output/narval/qwen3_1p7b_to_0p6b_hda/pilot/{NAME}/seed0'
DATA = 'data/narval/qwen3_1p7b_to_0p6b_hda'
OUTPUT = f'output/pd_diagnostics_{RUN_NAME}.json' if RUN_NAME else f'output/pd_diagnostics_{TAG}.json'
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/root/repo')

app = modal.App(APP_NAME)
# CRPD_DIAG_VOLUME=crpd-qwen3-pilot-hda-x30 measures a run of the 2,000-prompt pilot (same test prompts as x10).
VOLUME = os.environ.get('CRPD_DIAG_VOLUME','crpd-qwen3-pilot-hda-x10')
volume = modal.Volume.from_name(VOLUME)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0','transformers==4.52.4','numpy','safetensors','tokenizers','huggingface_hub')
         .env({'HF_HOME':'/vol/hf','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1',
               'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','CRPD_DIAG_TAG':TAG,
               'CRPD_DIAG_RUN':RUN_NAME,'CRPD_DIAG_VOLUME':VOLUME})
         .add_local_dir(REPO/'chunkratio','/root/repo/chunkratio',ignore=['__pycache__']))


def depth_bucket(n: int) -> str:
    """Response tokens before the state."""
    return 'n=0' if n == 0 else 'n<128' if n < 128 else 'n<256' if n < 256 else 'n>=256'


def kind(candidate: dict) -> str:
    """Proposing source of a candidate (both when rho and T+ drew the same chunk)."""
    sources = set(candidate['sources'])
    return 'both' if len(sources) > 1 else 'rho' if 'student_ref' in sources else 'teacher'


def repetition(tokens: list[int]) -> float:
    """Share of 4-grams that repeat an earlier 4-gram of the same response."""
    grams = [tuple(tokens[i:i+4]) for i in range(len(tokens)-3)]
    return 1-len(set(grams))/len(grams) if grams else 0.


@app.function(image=image,gpu='A100-80GB',cpu=4.0,memory=98304,timeout=2*3600,volumes={'/vol':volume})
def diagnose(teacher_pairs: int = 256) -> dict:
    import os
    import statistics
    import numpy as np
    import torch
    os.chdir('/vol')
    sys.path.insert(0,'/root/repo')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from chunkratio.artifacts import digest, read_json, read_jsonl
    from chunkratio.evaluation import reward_selector
    from chunkratio.pilot import HDA_CHECKPOINTS
    from chunkratio.scoring import continuation_increments_batch, length_groups, sibling_logprobs
    cfg = read_json(Path(RUN)/'resolved_config.json')
    eos_token = cfg['eos_token']
    def load(repo: str, revision: str | None = None) -> torch.nn.Module:
        """FP32 frozen model on the GPU."""
        model = AutoModelForCausalLM.from_pretrained(repo,torch_dtype=torch.float32,**({'revision':revision} if revision else {}))
        return model.to('cuda').eval().requires_grad_(False)
    rho = load(HDA_CHECKPOINTS['student_ref']['repo'],HDA_CHECKPOINTS['student_ref']['revision'])
    student = load(str(Path(RUN)/'student'))
    tokenizer = AutoTokenizer.from_pretrained(HDA_CHECKPOINTS['student_ref']['repo'],
                                              revision=HDA_CHECKPOINTS['student_ref']['revision'])
    tokenizer.eos_token = eos_token
    eos = tokenizer.convert_tokens_to_ids(eos_token)
    # Section A compares the student's margins with the frozen targets, so it scores chunks under the
    # run's own rule (score_end_token=false leaves the end-of-turn token unscored in both).
    score_end = cfg.get('score_end_token',True)
    report: dict = {'run':RUN,'pair_margin_end_token_scored':score_end,
                    'teacher_gap_scale':cfg.get('teacher_gap_scale',1.0)}

    # A. Pair targets by type and depth, with the trained student's margins (bf16 autocast, as trained).
    targets = read_json(Path(read_json(Path(RUN)/'target_index.json')['path']))
    pairs = targets['pairs']
    widths = [len(p['state']['prefix_ids'])+max(len(p['c']['tokens']),len(p['d']['tokens'])) for p in pairs]
    margins: list = [None]*len(pairs)
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        for group in length_groups(widths,16384):
            for start in range(0,len(group),48):
                part = group[start:start+48]
                prefixes = [pairs[i]['state']['prefix_ids'] for i in part]
                chunks = [[pairs[i]['c']['tokens'],pairs[i]['d']['tokens']] for i in part]
                s = sibling_logprobs(student,prefixes,chunks,eos,score_end)
                r = sibling_logprobs(rho,prefixes,chunks,eos,score_end)
                for i,a,b in zip(part,s,r):
                    margins[i] = (float(a[0]-b[0]),float(a[1]-b[1]))
    groups: dict[str,list] = {}
    for pair,(ac,ad) in zip(pairs,margins):
        t = pair['c']['target_score']-pair['d']['target_score']
        win,lose = (ac,ad) if t >= 0 else (ad,ac)
        types = '+'.join(sorted((kind(pair['c']),kind(pair['d']))))
        teacher_first = None
        if types == 'rho+teacher':
            teacher_side = 'c' if kind(pair['c']) == 'teacher' else 'd'
            teacher_first = (t >= 0) == (teacher_side == 'c')
        row = (pair['weight'],t,win,lose,(ac-ad)*t > 0,teacher_first)
        depth = depth_bucket(pair['state']['n'])
        for key in ('all',f'type:{types}',f'depth:{depth}',f'type:{types}|depth:{depth}'):
            groups.setdefault(key,[]).append(row)
    total = sum(p['weight'] for p in pairs)
    report['pairs'] = {}
    for key,rows in sorted(groups.items()):
        weight = sum(r[0] for r in rows)
        mixed = [r for r in rows if r[5] is not None]
        report['pairs'][key] = {'n':len(rows),'weight_share':weight/total,
                                'student_agrees_with_target':sum(r[0] for r in rows if r[4])/weight,
                                'mean_logratio_preferred':sum(r[0]*r[2] for r in rows)/weight,
                                'mean_logratio_dispreferred':sum(r[0]*r[3] for r in rows)/weight,
                                'mean_abs_target_gap':sum(r[0]*abs(r[1]) for r in rows)/weight,
                                'teacher_side_preferred':(sum(r[0] for r in mixed if r[5])/sum(r[0] for r in mixed)
                                                          if mixed else None)}
    print(json.dumps({'pairs_all':report['pairs']['all']}),flush=True)

    # B. Teacher preference on independent UltraFeedback pairs (none of the run's prompts).
    used = {r['id'] for r in read_jsonl(Path(cfg['prompts']))}
    pool = [r for r in read_jsonl(Path(DATA)/'ultrafeedback_train.jsonl')
            if r['id'] not in used and r.get('chosen_ids') and r.get('rejected_ids') and r['chosen_ids'] != r['rejected_ids']]
    pool = sorted(pool,key=lambda r:digest(['teacher-check',r['id']]))[:teacher_pairs]
    plus = load(HDA_CHECKPOINTS['teacher_plus']['repo'],HDA_CHECKPOINTS['teacher_plus']['revision'])
    base = load(HDA_CHECKPOINTS['teacher_base']['repo'],HDA_CHECKPOINTS['teacher_base']['revision'])
    def clip(ids: list[int]) -> list[int]:
        """Response up to and including its first EOS (scoring rule)."""
        return ids[:ids.index(eos)+1] if eos in ids else ids
    prefixes = [r['prompt_ids'] for r in pool for _ in range(2)]
    suffixes = [clip(r[k]) for r in pool for k in ('chosen_ids','rejected_ids')]
    lp_plus = continuation_increments_batch(plus,prefixes,suffixes,eos,8192)
    lp_base = continuation_increments_batch(base,prefixes,suffixes,eos,8192)
    rows = []
    for i in range(len(pool)):
        c,r = 2*i,2*i+1
        d_c = float((lp_plus[c].double()-lp_base[c].double()).sum())
        d_r = float((lp_plus[r].double()-lp_base[r].double()).sum())
        rows.append({'delta_sum':d_c-d_r,'delta_mean':d_c/len(suffixes[c])-d_r/len(suffixes[r]),
                     'plus_sum':float(lp_plus[c].double().sum()-lp_plus[r].double().sum()),
                     'length':len(suffixes[c])-len(suffixes[r])})
    share = lambda key: sum(row[key] > 0 for row in rows)/len(rows)
    report['teacher_preference'] = {'pairs':len(rows),'source':'ultrafeedback train_prefs rows outside the run prompts',
                                    'delta_sum_prefers_chosen':share('delta_sum'),
                                    'delta_per_token_prefers_chosen':share('delta_mean'),
                                    'teacher_plus_alone_prefers_chosen':share('plus_sum'),
                                    'longer_is_chosen':share('length'),
                                    'length_ties':sum(row['length'] == 0 for row in rows)/len(rows),
                                    'corr_delta_sum_with_length_difference':float(np.corrcoef(
                                        [r['delta_sum'] for r in rows],[r['length'] for r in rows])[0,1])}
    print(json.dumps({'teacher_preference':report['teacher_preference']}),flush=True)
    del plus,base
    torch.cuda.empty_cache()

    # C. Free generation on held-out test prompts (never used for selection).
    test = [{**r,'split':'select'} for r in read_jsonl(Path(DATA)/'ultrafeedback_test.jsonl')
            if len(r['prompt_ids']) <= cfg.get('max_prompt_tokens',2048)]
    selector,_ = reward_selector({**cfg,'selection_generation_batch_size':32},test,tokenizer)
    generations = {}
    for name,model in (('rho',rho),('student',student)):
        selector(model)
        generations[name] = {'scores':list(selector.last_scores),'responses':selector.last_responses}
    def drift(responses: list[list[int]]) -> float:
        """Mean per-token log pi - log rho on the given responses (FP32)."""
        pre = [r['prompt_ids'] for r,resp in zip(test,responses) if resp]
        suf = [resp for resp in responses if resp]
        a = continuation_increments_batch(student,pre,suf,eos,8192)
        b = continuation_increments_batch(rho,pre,suf,eos,8192)
        return sum(float((x.double()-y.double()).sum()) for x,y in zip(a,b))/sum(len(s) for s in suf)
    delta = np.asarray(generations['student']['scores'])-np.asarray(generations['rho']['scores'])
    boot = delta[np.random.default_rng(0).integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
    report['heldout_generation'] = {
        'prompts':len(test),
        'rm_mean':{k:statistics.mean(v['scores']) for k,v in generations.items()},
        'rm_student_minus_rho':{'mean':float(delta.mean()),'bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
                                'wins':int((delta > 0).sum()),'losses':int((delta < 0).sum())},
        'length_mean':{k:statistics.mean(len(r) for r in v['responses']) for k,v in generations.items()},
        'eos_rate':{k:sum(bool(r) and r[-1] == eos for r in v['responses'])/len(v['responses'])
                    for k,v in generations.items()},
        'repetition_4gram':{k:statistics.mean(repetition(r) for r in v['responses']) for k,v in generations.items()},
        'student_minus_rho_logratio_per_token':{f'on_{k}_responses':drift(v['responses']) for k,v in generations.items()}}
    # Per-prompt values and a length-controlled view of the RM gain (measurement only): the reward model may
    # favour longer answers, so the difference is also reported at equal length (OLS intercept of the reward
    # difference on the length difference) and on prompts where both responses ended with EOS.
    lengths = {k:[len(r) for r in v['responses']] for k,v in generations.items()}
    ended = {k:[bool(r) and r[-1] == eos for r in v['responses']] for k,v in generations.items()}
    dlen = np.asarray(lengths['student'],dtype=float)-np.asarray(lengths['rho'],dtype=float)
    slope,intercept = np.polyfit(dlen,delta,1) if np.ptp(dlen) > 0 else (0.,float(delta.mean()))
    both = [i for i in range(len(delta)) if ended['student'][i] and ended['rho'][i]]
    report['heldout_generation']['per_prompt'] = {'rm':{k:v['scores'] for k,v in generations.items()},
                                                  'length':lengths,'eos':ended}
    report['heldout_generation']['length_controlled'] = {
        'ols_rm_difference_at_equal_length':float(intercept),'ols_slope_per_token':float(slope),
        'corr_rm_difference_with_length_difference':float(np.corrcoef(dlen,delta)[0,1]) if np.ptp(dlen) > 0 else None,
        'both_eos_prompts':len(both),
        'both_eos_rm_difference':float(delta[both].mean()) if both else None,
        'both_eos_wins_losses':[int((delta[both] > 0).sum()),int((delta[both] < 0).sum())] if both else None}
    print(json.dumps({'heldout_generation':report['heldout_generation']}),flush=True)
    Path(OUTPUT).write_text(json.dumps(report,indent=1),encoding='utf-8')
    volume.commit()
    return report


if __name__ == '__main__':
    print('spawned',modal.Function.from_name(APP_NAME,'diagnose').spawn().object_id)


@app.function(image=image,gpu='A100-80GB',cpu=4.0,memory=65536,timeout=3600,volumes={'/vol':volume})
def eos_check(prompts: int = 32, endings: int = 64) -> dict:
    """Which end-of-turn token each role predicts: at rho's own response ends (<|im_end|> vs <|endoftext|>
    probabilities under T+, T0 and rho), and in each role's greedy generations on held-out prompts."""
    import os
    import statistics
    import torch
    os.chdir('/vol')
    sys.path.insert(0,'/root/repo')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from chunkratio.artifacts import read_json, read_jsonl
    from chunkratio.pilot import HDA_CHECKPOINTS
    tok = AutoTokenizer.from_pretrained(HDA_CHECKPOINTS['student_ref']['repo'],revision=HDA_CHECKPOINTS['student_ref']['revision'])
    im_end, endoftext = tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>')
    prefixes = sorted(Path(read_json(Path(RUN)/'target_index.json')['path']).parent.parent.glob('prefixes-*'))[0]
    ends = []
    for path in sorted(prefixes.glob('*.jsonl')):
        payload = read_json(path)['payload']
        if payload['response'] and payload['response'][-1] == im_end:
            ends.append(payload['prompt']['prompt_ids']+payload['response'][:-1])
        if len(ends) >= endings:
            break
    test = read_jsonl(Path(DATA)/'ultrafeedback_test.jsonl')[:prompts]
    report = {'im_end':im_end,'endoftext':endoftext,'rho_endings':len(ends)}
    for role in ('teacher_plus','teacher_base','student_ref'):
        source = HDA_CHECKPOINTS[role]
        model = AutoModelForCausalLM.from_pretrained(source['repo'],revision=source['revision'],torch_dtype=torch.float32)
        model = model.to('cuda').eval().requires_grad_(False)
        lp_im, lp_eot = [], []
        with torch.no_grad():
            for ids in ends:
                logits = model(torch.tensor([ids],device='cuda')).logits[0,-1].float().log_softmax(-1)
                lp_im.append(float(logits[im_end])); lp_eot.append(float(logits[endoftext]))
            stops, lengths = {'im_end':0,'endoftext':0,'none':0}, []
            for row in test:
                ids = torch.tensor([row['prompt_ids']],device='cuda')
                out = model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=1024,do_sample=False,
                                     eos_token_id=[im_end,endoftext],pad_token_id=endoftext)[0,ids.shape[1]:].tolist()
                lengths.append(len(out))
                stops['im_end' if out and out[-1] == im_end else 'endoftext' if out and out[-1] == endoftext else 'none'] += 1
        report[role] = {'logp_im_end_at_rho_end':statistics.mean(lp_im),'logp_endoftext_at_rho_end':statistics.mean(lp_eot),
                        'greedy_stop_token':stops,'greedy_length_mean':statistics.mean(lengths)}
        print(json.dumps({role:report[role]}),flush=True)
        del model
        torch.cuda.empty_cache()
    Path('output/eos_check_hda.json').write_text(json.dumps(report,indent=1),encoding='utf-8')
    volume.commit()
    return report
