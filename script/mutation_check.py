"""Review mutation check: apply each source mutation, run pytest, report KILLED/SURVIVED, always revert.

Run from the repository root: python script/mutation_check.py [mutation names...]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
MUTATIONS = {
    "flip_delta_T_sign": ("chunkratio/math.py",
        "return (stable(logp_plus) - stable(logp_base)",
        "return (stable(logp_base) - stable(logp_plus)"),
    "drop_mu_hat": ("chunkratio/math.py",
        "\n            + stable(mu_hat).masked_fill(terminal, 0)).detach()",
        ").detach()"),
    "target_terminal_mask_removed": ("chunkratio/math.py",
        "stable(mu_hat).masked_fill(terminal, 0)", "stable(mu_hat)"),
    "alpha_outside_sigmoid": ("chunkratio/math.py",
        "F.binary_cross_entropy_with_logits(margin, u.sigmoid(), reduction=\"none\")",
        "F.binary_cross_entropy_with_logits(margin, alpha*(u/alpha).sigmoid(), reduction=\"none\")"),
    "prompt_tokens_in_mask": ("chunkratio/scoring.py",
        "response = (positions >= prompt_lengths[:,None])",
        "response = (positions >= 1)"),
    "remove_no_grad_score_batch": ("chunkratio/scoring.py",
        "with torch.no_grad() if frozen else contextlib.nullcontext():",
        "with contextlib.nullcontext():"),
    "remove_no_grad_increments": ("chunkratio/scoring.py",
        "@torch.no_grad()\ndef continuation_increments", "def continuation_increments"),
    "remove_no_grad_features": ("chunkratio/scoring.py",
        "@torch.no_grad()\ndef state_features", "def state_features"),
    "remove_no_grad_generate": ("chunkratio/sampling.py",
        "@torch.no_grad()\ndef generate_batch", "def generate_batch"),
    "forward_cumsum_returns": ("chunkratio/math.py",
        "return stable(increments).flip((-1,)).cumsum(-1).flip((-1,))",
        "return stable(increments).cumsum(-1)"),
    "terminal_mu_not_zeroed": ("chunkratio/targets.py",
        "cfg['variant'] in ('crpd_raw','crpd_suffix') or row['terminal']",
        "cfg['variant'] in ('crpd_raw','crpd_suffix')"),
    "logit_label_off_by_one": ("chunkratio/scoring.py",
        "raw = logits[:,:-1,:]", "raw = logits[:,1:,:]"),
    "eos_excluded": ("chunkratio/scoring.py",
        "mask = response & (seen_before == 0)", "mask = response & (seen_before == 0) & ~eos_hits"),
    "post_eos_included": ("chunkratio/scoring.py",
        "mask = response & (seen_before == 0)", "mask = response"),
    "ref_sign_in_margin": ("chunkratio/math.py",
        "(stable(logp_student_c) - stable(logp_ref_c.detach()))",
        "(stable(logp_student_c) + stable(logp_ref_c.detach()))"),
    "target_not_detached": ("chunkratio/math.py",
        "return (alpha * (stable(t_c) - stable(t_d))).detach()",
        "return (alpha * (stable(t_c) - stable(t_d)))"),
    "routing_order_swapped": ("chunkratio/math.py",
        "    if teacher_plus_id == teacher_base_id:\n        return \"no_alignment_update\"\n    if student_ref_id == teacher_base_id:\n        return \"exact_zero_residual\"",
        "    if student_ref_id == teacher_base_id:\n        return \"exact_zero_residual\"\n    if teacher_plus_id == teacher_base_id:\n        return \"no_alignment_update\""),
    "flat_pair_weights": ("chunkratio/targets.py",
        "'weight':1/len(grouped)/len(prompt_states)/math.comb(len(cs),2)})",
        "'weight':1.0})\n    for p in pairs: p['weight'] = 1/len(pairs)\n    if False: pass"),
    "flat_pair_sampler": ("chunkratio/training.py",
        "states = self.groups[self.rng.choice(list(self.groups))]\n            batch.append(self.rng.choice(states[self.rng.choice(list(states))]))",
        "batch.append(self.rng.choice(self.targets['pairs']))"),
    "head_divides_by_length": ("chunkratio/math.py",
        ".square().sum() / horizon)", ".square().sum() / len(pred))"),
    "correction_from_teacher": ("chunkratio/collection.py",
        "source = 'teacher_plus' if teacher_direction else 'student_ref'",
        "source = 'teacher_plus'"),
    "budget_reset_at_chunk": ("chunkratio/collection.py",
        "[cfg['horizon']-state['n'] for state,_ in jobs],eos,rngs,length)",
        "[cfg['horizon'] for state,_ in jobs],eos,rngs,length)"),
    "alpha_missing_on_margin": ("chunkratio/math.py",
        "margin = alpha * ((stable(logp_student_c)", "margin = ((stable(logp_student_c)"),
    "student_detached": ("chunkratio/math.py",
        "(stable(logp_student_c) - stable(logp_ref_c.detach()))",
        "(stable(logp_student_c.detach()) - stable(logp_ref_c.detach()))"),
    "trajectory_feature_off_by_one": ("chunkratio/scoring.py",
        "ids = torch.tensor([prefix+tokens[:-1]],device=next(model.parameters()).device)\n    hidden = model(ids,attention_mask=torch.ones_like(ids),output_hidden_states=True,use_cache=False).hidden_states[-1]\n    return hidden[0,len(prefix)-1:]",
        "ids = torch.tensor([prefix+tokens],device=next(model.parameters()).device)\n    hidden = model(ids,attention_mask=torch.ones_like(ids),output_hidden_states=True,use_cache=False).hidden_states[-1]\n    return hidden[0,len(prefix):]"),
    "correction_budget_ignores_n": ("chunkratio/collection.py",
        "budgets = [(cfg['horizon']-seed['n']-t)/cfg['horizon']", "budgets = [(cfg['horizon']-t)/cfg['horizon']"),
    "endpoint_budget_ignores_chunk": ("chunkratio/targets.py",
        "budget = torch.tensor([(cfg['horizon']-state['n']-len(row['tokens']))/cfg['horizon']]",
        "budget = torch.tensor([(cfg['horizon']-state['n'])/cfg['horizon']]"),
    "no_alignment_still_steps": ("chunkratio/training.py",
        "if self.targets['route'] == 'no_alignment_update' or cfg['variant'] == 'crpd_audit':",
        "if cfg['variant'] == 'crpd_audit':"),
    "exact_zero_uses_head": ("chunkratio/targets.py",
        "bypass = route != 'reference_mean' or", "bypass = route == 'no_alignment_update' or"),
    "shared_collection_broken": ("chunkratio/config.py",
        "return cfg.get('collection_seed',0)", "return cfg.get('collection_seed',0)+cfg['seed']"),
    "state_selection_reads_outcome": ("chunkratio/collection.py",
        "selected = select_states(positions,prompt['id'],seed,cfg.get('states_per_prompt'))",
        "selected = select_states(positions,prompt['id']+digest(response),seed,cfg.get('states_per_prompt'))"),
    "per_row_generator_removed": ("chunkratio/sampling.py",
        "generator=rngs[i]", "generator=rngs[0]"),
    "eos_from_pad_token": ("chunkratio/scoring.py",
        "'eos':signatures[0]['eos']", "'eos':signatures[0]['pad']"),
    "eos_check_skipped": ("chunkratio/scoring.py",
        "if eos_token is not None and tok.get_vocab().get(eos_token) != tok.eos_token_id:", "if False:"),
}


def main() -> None:
    """Run the selected mutations (default all) and print one result line each."""
    names = sys.argv[1:] or list(MUTATIONS)
    results = {}
    for name in names:
        path, old, new = MUTATIONS[name]
        file = ROOT / path
        original = file.read_bytes()
        text = original.decode()
        assert text.count(old) == 1, (name, text.count(old))
        try:
            file.write_bytes(text.replace(old, new).encode())
            run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-W", "ignore::pytest.PytestConfigWarning"],
                                 capture_output=True, text=True)
            last = run.stdout.strip().splitlines()[-1]
            results[name] = {"killed": run.returncode != 0, "summary": last}
        finally:
            file.write_bytes(original)
        print(name, "KILLED" if results[name]["killed"] else "SURVIVED", "|", results[name]["summary"], flush=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
