"""Held-out generation check of a finished CRPD run (measurement only; never used for selection).

Same rule as section C of script/modal/pd_diagnostics.py: greedy responses (up to 1024 new tokens) of rho and the
trained student on the prepared test prompts, pinned FsfairX RM reward with a paired bootstrap, EOS rate, length,
4-gram repetition, the student-rho per-token log-ratio on each model's responses, per-prompt values and the
length-controlled RM difference. Writes <run>/heldout.json.

  python script/server/heldout_eval.py --run output/server/<pair>/<scale>/runs/<name>/seed0 --test <test.jsonl>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def repetition(tokens: list[int]) -> float:
    """Share of 4-grams that repeat an earlier 4-gram of the same response."""
    grams = [tuple(tokens[i:i+4]) for i in range(len(tokens)-3)]
    return 1-len(set(grams))/len(grams) if grams else 0.


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True,help='Finished run folder (holds resolved_config.json and student/)')
    parser.add_argument('--test',required=True,help='Prepared test prompts (JSONL with messages)')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--limit',type=int,default=None,help='First N test prompts only (smoke)')
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from chunkratio.artifacts import read_json, read_jsonl, write_json
    from chunkratio.evaluation import reward_selector
    from chunkratio.scoring import continuation_increments_batch
    run = Path(args.run)
    cfg = {**read_json(run/'resolved_config.json'),'device':args.device}
    rho_spec = cfg['models']['student_ref']
    def load(path: str, revision: str | None = None) -> torch.nn.Module:
        """FP32 frozen model on the device."""
        model = AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float32,**({'revision':revision} if revision else {}))
        return model.to(args.device).eval().requires_grad_(False)
    rho = load(rho_spec['path'],rho_spec['revision'])
    student = load(str(run/'student'))
    tokenizer = AutoTokenizer.from_pretrained(rho_spec['path'],revision=rho_spec['revision'])
    tokenizer.eos_token = cfg['eos_token']
    eos = tokenizer.convert_tokens_to_ids(cfg['eos_token'])
    test = [{**r,'split':'select'} for r in read_jsonl(Path(args.test))
            if len(r['prompt_ids']) <= cfg.get('max_prompt_tokens',2048)]
    test = test[:args.limit] if args.limit else test
    selector,cost = reward_selector({**cfg,'selection_generation_batch_size':32},test,tokenizer)
    generations = {}
    for name,model in (('rho',rho),('student',student)):
        selector(model)
        generations[name] = {'scores':list(selector.last_scores),'responses':selector.last_responses}
    def drift(responses: list[list[int]]) -> float:
        """Mean per-token log student - log rho on the given responses (FP32)."""
        pre = [r['prompt_ids'] for r,resp in zip(test,responses) if resp]
        suf = [resp for resp in responses if resp]
        a = continuation_increments_batch(student,pre,suf,eos,8192)
        b = continuation_increments_batch(rho,pre,suf,eos,8192)
        return sum(float((x.double()-y.double()).sum()) for x,y in zip(a,b))/sum(len(s) for s in suf)
    delta = np.asarray(generations['student']['scores'])-np.asarray(generations['rho']['scores'])
    boot = delta[np.random.default_rng(0).integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
    lengths = {k:[len(r) for r in v['responses']] for k,v in generations.items()}
    ended = {k:[bool(r) and r[-1] == eos for r in v['responses']] for k,v in generations.items()}
    dlen = np.asarray(lengths['student'],dtype=float)-np.asarray(lengths['rho'],dtype=float)
    slope,intercept = np.polyfit(dlen,delta,1) if np.ptp(dlen) > 0 else (0.,float(delta.mean()))
    both = [i for i in range(len(delta)) if ended['student'][i] and ended['rho'][i]]
    report = {
        'run':str(run),'prompts':len(test),
        'rm_mean':{k:statistics.mean(v['scores']) for k,v in generations.items()},
        'rm_student_minus_rho':{'mean':float(delta.mean()),'bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
                                'wins':int((delta > 0).sum()),'losses':int((delta < 0).sum())},
        'length_mean':{k:statistics.mean(v) for k,v in lengths.items()},
        'eos_rate':{k:sum(v)/len(v) for k,v in ended.items()},
        'repetition_4gram':{k:statistics.mean(repetition(r) for r in v['responses']) for k,v in generations.items()},
        'student_minus_rho_logratio_per_token':{f'on_{k}_responses':drift(v['responses']) for k,v in generations.items()},
        'length_controlled':{'ols_rm_difference_at_equal_length':float(intercept),'ols_slope_per_token':float(slope),
                             'both_eos_prompts':len(both),
                             'both_eos_rm_difference':float(delta[both].mean()) if both else None,
                             'both_eos_wins_losses':[int((delta[both] > 0).sum()),int((delta[both] < 0).sum())] if both else None},
        'per_prompt':{'rm':{k:v['scores'] for k,v in generations.items()},'length':lengths,'eos':ended},
        'cost':cost}
    write_json(run/'heldout.json',report)
    print(json.dumps({k:report[k] for k in ('rm_student_minus_rho','eos_rate','length_mean','repetition_4gram','length_controlled')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
