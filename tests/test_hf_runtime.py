"""Non-synthetic loading/scoring path on tiny local Llama checkpoints, no downloads (§§3,5,6,12)."""
import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch',reason='HF runtime requires PyTorch')
transformers = pytest.importorskip('transformers',reason='HF runtime requires transformers')
from chunkratio.artifacts import read_json
from chunkratio.config import load_config
from chunkratio.pipeline import pipeline
from chunkratio.scoring import score_batch, state_features, trajectory_features

TEMPLATE = ("{% for m in messages %}<bos>{{ m['content'] }}{% if m['role'] == 'assistant' %}<eos>{% endif %}"
            "{% endfor %}{% if add_generation_prompt %}<bos>{% endif %}")


def tokenizer() -> object:
    """Fast word-level tokenizer with explicit EOS and chat template (§5.1)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {'<eos>':0,'<bos>':1,**{f'w{i}':i for i in range(2,16)}}
    core = Tokenizer(models.WordLevel(vocab,unk_token='<eos>'))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core,eos_token='<eos>',bos_token='<bos>',
                                               model_input_names=['input_ids','attention_mask'])
    tok.chat_template = TEMPLATE
    return tok


def checkpoints(root: Path) -> dict[str,Path]:
    """T0, perturbed T+ and a distinct rho, all sharing one tokenizer (§12)."""
    config = transformers.LlamaConfig(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                                      num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,
                                      eos_token_id=0,bos_token_id=1,pad_token_id=None)
    torch.manual_seed(0)
    base = transformers.LlamaForCausalLM(config)
    plus = copy.deepcopy(base)
    with torch.no_grad():
        for p in plus.parameters():
            p.add_(torch.randn_like(p)*.05)
    torch.manual_seed(1)
    rho = transformers.LlamaForCausalLM(config)
    paths = {}
    for name,model in (('teacher_base',base),('teacher_plus',plus),('student_ref',rho)):
        paths[name] = root/name
        model.save_pretrained(paths[name])
        tokenizer().save_pretrained(paths[name])
    return paths


def real_config(tmp_path: Path, paths: dict[str,Path]) -> dict:
    """Resolved non-synthetic config pointing at local checkpoints (§11)."""
    prompts = tmp_path/'prompts.jsonl'
    rows = [{'prompt_ids':[1,2+i,3+i,1],'messages':[{'role':'user','content':f'w{2+i} w{3+i}'}]} for i in range(12)]
    prompts.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    cfg = load_config('config/crpd/smoke.json')
    cfg.update(synthetic=False,prompts=str(prompts),output=str(tmp_path/'run'),teacher_dpo_reference_verified=True,
               selection={'reward_model_revision':'pinned','decoding':{'max_new_tokens':8,'do_sample':False}},
               selection_interval=1,models={k:{'path':str(v),'revision':'local'} for k,v in paths.items()})
    return cfg


def test_hf_llama_scoring_padding_and_features(tmp_path: Path) -> None:
    """Right padding with EOS=PAD, causal features and one-pass trajectory features on Llama (§6)."""
    paths = checkpoints(tmp_path)
    model = transformers.LlamaForCausalLM.from_pretrained(paths['student_ref']).eval()
    prefixes, chunks = [[1,2,3],[1,4]], [[5,0],[6,7,8]]
    batched = score_batch(model,prefixes,chunks,0)
    for i in range(2):
        torch.testing.assert_close(batched[i:i+1],score_batch(model,[prefixes[i]],[chunks[i]],0),atol=1e-5,rtol=0)
    trajectory = trajectory_features(model,[1,2,3],[5,6,0])
    for t,state in enumerate([[1,2,3],[1,2,3,5],[1,2,3,5,6]]):
        torch.testing.assert_close(trajectory[t],state_features(model,[state])[0],atol=1e-5,rtol=0)
    padded = state_features(model,[[1,2,3],[1,2,3,5,6,7]])
    torch.testing.assert_close(padded[0],state_features(model,[[1,2,3]])[0],atol=1e-5,rtol=0)


def test_hf_pipeline_routes_through_targets(tmp_path: Path) -> None:
    """Non-synthetic load, preflight, routing and stages to frozen targets on real HF classes (§§4.1,10)."""
    paths = checkpoints(tmp_path)
    cfg = real_config(tmp_path,paths)
    result = pipeline(cfg,'build-targets')
    assert result['route'] == 'reference_mean' and result['pairs'] > 0
    assert result['head']['hidden_size'] == 16
    manifest = read_json(tmp_path/'run'/'manifest.json')
    assert manifest['synthetic'] is False and manifest['tokenizer']['eos'] == 0
    shared = {**cfg,'output':str(tmp_path/'shared'),
              'models':{**cfg['models'],'student_ref':cfg['models']['teacher_base']}}
    shared_result = pipeline(shared,'build-targets')
    assert shared_result['route'] == 'exact_zero_residual' and shared_result['head'] == {'hash':'exact-zero'}
    null = {**cfg,'output':str(tmp_path/'null'),'models':{**cfg['models'],'teacher_plus':cfg['models']['teacher_base']}}
    assert pipeline(null,'evaluate') == {'route':'no_alignment_update','optimizer_steps':0,'collection_skipped':True}
    exported = transformers.LlamaForCausalLM.from_pretrained(tmp_path/'null'/'student')
    original = transformers.LlamaForCausalLM.from_pretrained(paths['student_ref'])
    for a,b in zip(exported.state_dict().values(),original.state_dict().values()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_hf_policy_selection_export_and_rm_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-path RM checkpoint selection on select prompts, export, and held-out RM scoring (§§8,11.1,11.4)."""
    from chunkratio.evaluation import evaluate_reward_model
    paths = checkpoints(tmp_path)
    rm_config = transformers.LlamaConfig(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                                         num_attention_heads=2,num_key_value_heads=2,num_labels=1,
                                         max_position_embeddings=64,pad_token_id=0)
    torch.manual_seed(2)
    reward = transformers.LlamaForSequenceClassification(rm_config)
    seen = []
    real_tokenizer = transformers.AutoTokenizer.from_pretrained
    def rm_model(name: str, **kwargs: object) -> object:
        seen.append((name,kwargs['revision']))
        return copy.deepcopy(reward)
    def any_tokenizer(name: object, **kwargs: object) -> object:
        return tokenizer() if name == 'sfairXC/FsfairX-LLaMA3-RM-v0.1' else real_tokenizer(name,**kwargs)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification,'from_pretrained',rm_model)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',any_tokenizer)
    cfg = real_config(tmp_path,paths)
    rows = [{**json.loads(line),'split':['select','head_val','test','train','train','train'][i%6]}
            for i,line in enumerate(Path(cfg['prompts']).read_text(encoding='utf-8').splitlines())]
    Path(cfg['prompts']).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    cfg.update(steps=2,selection_interval=1)
    result = pipeline(cfg,'evaluate')
    assert result['policy']['selection_rule'] == 'select_mean_reward'
    assert result['policy']['optimizer_steps'] == 2
    assert all(h['selection_score'] is not None for h in result['policy']['history'])
    assert result['cost']['selection_this_invocation']['selection_calls'] == 2
    assert seen and all(revision == 'pinned' for _,revision in seen)
    summary = evaluate_reward_model(cfg,str(tmp_path/'run'/'student'),tmp_path/'rm.jsonl')
    assert summary['n_prompts'] == 2 and summary['split'] == 'test'


def test_preparation_sft_then_teacher_dpo(tmp_path: Path) -> None:
    """SFT then DPO launchers run; DPO records its frozen reference = the SFT checkpoint (§11.1)."""
    from chunkratio.preparation import prepare_checkpoint
    from chunkratio.scoring import fingerprint
    paths = checkpoints(tmp_path)
    data = tmp_path/'pairs.jsonl'
    rows = [{'prompt_ids':[1,2+i],'chosen_ids':[5,6,0],'rejected_ids':[7,0],'split':'train'} for i in range(3)]
    data.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    common = {'revision':'local','data':str(data),'device':'cpu','seed':0,'batch_size':128}
    prepare_checkpoint({**common,'model':str(paths['teacher_base']),'output':str(tmp_path/'sft'),
                        'epochs':3,'learning_rate':2e-5},'train-sft')
    sft = read_json(tmp_path/'sft'/'preparation_manifest.json')
    assert len(sft['history']) == 3
    prepare_checkpoint({**common,'model':str(tmp_path/'sft'),'output':str(tmp_path/'dpo'),
                        'epochs':2,'learning_rate':5e-7,'beta':.05},'train-teacher-dpo')
    dpo = read_json(tmp_path/'dpo'/'preparation_manifest.json')
    reference = transformers.LlamaForCausalLM.from_pretrained(tmp_path/'sft').eval().requires_grad_(False)
    assert dpo['teacher_dpo_reference_fingerprint'] == fingerprint(reference) == sft['model_fingerprint']
    assert dpo['model_fingerprint'] != sft['model_fingerprint']
    with pytest.raises(ValueError,match='beta'):
        prepare_checkpoint({**common,'model':str(tmp_path/'sft'),'output':str(tmp_path/'x'),
                            'epochs':2,'learning_rate':5e-7,'beta':.1},'train-teacher-dpo')
