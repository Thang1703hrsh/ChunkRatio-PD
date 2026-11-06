"""Modal pilot launcher: stage graph and GPU placement (runs only where `modal` is installed)."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('modal',reason='modal client not installed in this interpreter')


def test_launcher_stages_and_hardware() -> None:
    """GPUs only on training/scoring stages; orchestrators and data preparation are CPU-only."""
    spec = importlib.util.spec_from_file_location('qwen3_pilot',Path('script/modal/qwen3_pilot.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = Path('script/modal/qwen3_pilot.py').read_text(encoding='utf-8')
    for name in ('prepare','sft_teacher','sft_student','dpo_teacher','diagnose','train_variants','finish','run_pilot'):
        assert hasattr(module,name)
    assert "@app.function(cpu=2.0,memory=12288,**common)  # (3.7 GB smoke)\ndef prepare" in source
    assert "@app.function(cpu=0.25,memory=1024,**common)\ndef run_pilot" in source
    assert "gpu='L40S',cpu=2.0,memory=12288,retries=0,**sft_limits)" in source
    assert "\ndef sft_student" in source and "gpu='A100-80GB',cpu=2.0,memory=24576,retries=0,**dpo_limits)" in source
    assert "\ndef dpo_teacher" in source and module.SFT_DATA.endswith('fit2048.jsonl')
    assert module.VARIANTS == ('crpd_main','crpd_raw','crpd_anchor')


def test_smoke_scale_and_hf_push_switches(monkeypatch) -> None:
    """Smoke has its own app/volume and HF repo prefix; pushing needs the huggingface secret."""
    def load(scale: str, push: str):
        """Import the launcher under one environment."""
        monkeypatch.setenv('CRPD_SCALE',scale)
        monkeypatch.setenv('CRPD_HF_PUSH',push)
        spec = importlib.util.spec_from_file_location(f'qwen3_pilot_{scale}_{push}',Path('script/modal/qwen3_pilot.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    smoke = load('smoke','1')
    assert smoke.APP_NAME == 'crpd-qwen3-pilot-smoke' and smoke.HF_PREFIX.endswith('-smoke')
    assert smoke.VARIANTS == ('crpd_main',) and smoke.SMOKE_SIZES['subset'] == {'train':8,'head_val':4,'select':4}
    assert smoke.hf_secret and hasattr(smoke,'push_hf') and hasattr(smoke,'push_all')
    full = load('full','0')
    assert full.APP_NAME == 'crpd-qwen3-pilot' and not full.hf_secret and len(full.VARIANTS) == 3
    with pytest.raises(SystemExit,match='CRPD_SCALE'):
        load('huge','0')
    monkeypatch.setenv('CRPD_TEACHERS','hub')
    hub = load('full','1')
    assert hub.APP_NAME == 'crpd-qwen3-pilot-hub' and hub.VARIANTS == ('crpd_main','crpd_raw')
    assert hub.ROOT == 'output/narval/qwen3_1p7b_to_0p6b_hub/pilot' and hub.SFT_DATA.endswith('ultrachat_sft_fit2048.jsonl')
    assert hub.SIZES['states_per_prompt'] == 2 and hasattr(hub,'verify_teachers')
    stability = load('stability','1')
    assert stability.APP_NAME == 'crpd-qwen3-pilot-hub-stability' and stability.VARIANTS == ('crpd_main',)
    assert (stability.SIZES['steps'],stability.SIZES['selection_interval'],stability.SIZES['states_per_prompt']) == (150,50,2)
    assert stability.HF_PREFIX.endswith('-hub-stability') and stability.GPU_TAG == ''
    assert stability.SIZES['policy_microbatch_size'] == hub.SIZES['policy_microbatch_size'] == 4
    assert stability.SIZES['selection_baseline'] and 'policy_microbatch_size' not in stability.SMOKE_SIZES
    x10 = load('x10','0')
    assert x10.APP_NAME == 'crpd-qwen3-pilot-hub-x10' and x10.VARIANTS == ('crpd_main',)
    assert x10.SIZES['subset'] == {'train':640,'head_val':160,'select':64} and x10.SIZES['max_prompt_tokens'] == 2048
    assert x10.SIZES['policy_microbatch_size'] == 4 and x10.SIZES['uf_train_rows'] == 10*stability.SIZES['uf_train_rows']
    monkeypatch.setenv('CRPD_GPU','L40S')
    l40s = load('smoke','0')
    assert l40s.GPU_TAG == '-l40s' and l40s.CRPD_GPU == 'L40S'
    assert "'CRPD_GPU':CRPD_GPU" in Path('script/modal/qwen3_pilot.py').read_text(encoding='utf-8')
    monkeypatch.delenv('CRPD_GPU')
    monkeypatch.setenv('CRPD_TEACHERS','pretrained')
    with pytest.raises(SystemExit,match='CRPD_TEACHERS'):
        load('full','0')
    monkeypatch.delenv('CRPD_TEACHERS')
    assert full.ALPHAS == ()
    monkeypatch.setenv('CRPD_ALPHAS','0.03,0.1')
    sweep = load('stability','0')
    assert sweep.ALPHAS == (.03,.1) and sweep.run_name('crpd_main',.03) == 'crpd_main-alpha0p03'
    assert sweep.APP_NAME == 'crpd-qwen3-pilot-stability'
    assert "'CRPD_ALPHAS':ALPHA_TEXT" in Path('script/modal/qwen3_pilot.py').read_text(encoding='utf-8')
    monkeypatch.setenv('CRPD_ALPHAS','0,0.1')
    with pytest.raises(SystemExit,match='CRPD_ALPHAS'):
        load('stability','0')
    monkeypatch.delenv('CRPD_ALPHAS')
    assert not sweep.SWEEP_LITE
    monkeypatch.setenv('CRPD_SWEEP_LITE','1')
    assert load('x10','0').SWEEP_LITE
    monkeypatch.delenv('CRPD_SWEEP_LITE')
    monkeypatch.setenv('CRPD_VARIANTS','crpd_raw')
    assert load('x10','0').VARIANTS == ('crpd_raw',)
    monkeypatch.delenv('CRPD_VARIANTS')


@pytest.mark.parametrize('pair',['qwen3_8b_to_1p7b','llama31_8b_to_llama32_1b'])
def test_pair_launcher_hardware_and_stages(pair: str, monkeypatch) -> None:
    """8B stages on B200, one app/volume per pair, gated Llama needs the huggingface secret."""
    monkeypatch.setenv('CRPD_PAIR',pair)
    spec = importlib.util.spec_from_file_location(f'narval_pair_{pair}',Path('script/modal/narval_pair.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('prepare','sft_teacher','sft_student','dpo_teacher','diagnose','train_variants','finish','run_pilot'):
        assert hasattr(module,name)
    assert module.APP_NAME == 'crpd-'+pair.replace('_','-') and module.SFT_DATA.endswith('fit2048.jsonl')
    hardware = module.HARDWARE[pair]
    assert hardware['sft_teacher'][0] == hardware['dpo_teacher'][0] == hardware['variants'][0] == 'B200'
    assert module.stage('sft_teacher')['retries'] == 0 and module.stage('variants',retries=1)['retries'] == 1
    assert bool(module.secrets) == pair.startswith('llama')


def test_pair_launcher_requires_a_known_pair(monkeypatch) -> None:
    """No silent default: an unset or default pair stops before any Modal object is built."""
    for value in ('','qwen3_1p7b_to_0p6b'):
        monkeypatch.setenv('CRPD_PAIR',value)
        spec = importlib.util.spec_from_file_location('narval_pair_bad',Path('script/modal/narval_pair.py'))
        with pytest.raises(SystemExit,match='CRPD_PAIR'):
            spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_launcher_imports_inside_container(tmp_path,monkeypatch) -> None:
    """Modal re-imports the file as /root/qwen3_pilot.py, where parents[2] does not exist."""
    import modal
    shallow = tmp_path/'qwen3_pilot.py'
    shallow.write_text(Path('script/modal/qwen3_pilot.py').read_text(encoding='utf-8'),encoding='utf-8')
    monkeypatch.setattr(modal,'is_local',lambda: False)
    spec = importlib.util.spec_from_file_location('qwen3_pilot_remote',shallow)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.REPO == Path(module.REMOTE_REPO)


def test_finished_needs_same_config_and_same_code(tmp_path, monkeypatch) -> None:
    """A run of the same config but older code is not finished; another config's results are an error."""
    import json
    from chunkratio.artifacts import code_revision
    spec = importlib.util.spec_from_file_location('qwen3_pilot_finished',Path('script/modal/qwen3_pilot.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'workdir',lambda: tmp_path)
    config = {'output':'out/run','alpha':.3}
    out = tmp_path/'out'/'run'
    out.mkdir(parents=True)
    assert not module.finished(config,'diagnostics.json','abc')
    (out/'diagnostics.json').write_text('{}',encoding='utf-8')
    (out/'resolved_config.json').write_text(json.dumps(config),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'code_revision':'abc+src.0000000000000000'}),encoding='utf-8')
    assert not module.finished(config,'diagnostics.json','abc')
    (out/'manifest.json').write_text(json.dumps({'code_revision':code_revision('abc')}),encoding='utf-8')
    assert module.finished(config,'diagnostics.json','abc') and not module.finished(config,'diagnostics.json','def')
    with pytest.raises(ValueError,match='different config'):
        module.finished({**config,'alpha':.1},'diagnostics.json','abc')


def test_gamma_sweep_names_and_guards(monkeypatch) -> None:
    """CRPD_GAMMAS: gamma = 1 keeps the plain alpha name; gamma < 1 needs alpha and crpd_raw only (DECISIONS)."""
    def load(**env):
        """Import the launcher under one environment."""
        for key in ('CRPD_ALPHAS','CRPD_GAMMAS','CRPD_VARIANTS'):
            monkeypatch.delenv(key,raising=False)
        for key,value in env.items():
            monkeypatch.setenv(key,value)
        monkeypatch.setenv('CRPD_SCALE','x10')
        monkeypatch.setenv('CRPD_TEACHERS','hda')
        spec = importlib.util.spec_from_file_location('qwen3_pilot_gamma',Path('script/modal/qwen3_pilot.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    sweep = load(CRPD_ALPHAS='0.03',CRPD_GAMMAS='1,0.3,0.1',CRPD_VARIANTS='crpd_raw')
    assert sweep.GAMMAS == (1.,.3,.1)
    assert sweep.run_name('crpd_raw',.03) == sweep.run_name('crpd_raw',.03,1.) == 'crpd_raw-alpha0p03'
    assert sweep.run_name('crpd_raw',.03,.3) == 'crpd_raw-alpha0p03-gamma0p3'
    assert "'CRPD_GAMMAS':GAMMA_TEXT" in Path('script/modal/qwen3_pilot.py').read_text(encoding='utf-8')
    with pytest.raises(SystemExit,match='crpd_raw only'):
        load(CRPD_ALPHAS='0.03',CRPD_GAMMAS='0.3',CRPD_VARIANTS='crpd_main,crpd_raw')
    with pytest.raises(SystemExit,match='needs CRPD_ALPHAS'):
        load(CRPD_GAMMAS='0.3',CRPD_VARIANTS='crpd_raw')
    with pytest.raises(SystemExit,match=r'in \(0, 1\]'):
        load(CRPD_ALPHAS='0.03',CRPD_GAMMAS='0,0.3',CRPD_VARIANTS='crpd_raw')


def test_cache_tag_is_independent_of_output_tag(monkeypatch) -> None:
    """CRPD_CACHE_TAG names the cache apart from the outputs; unset, the cache follows the output tag as
    before; the tag reaches the container (DECISIONS 87)."""
    def load(**env):
        """Import the launcher under one environment."""
        for key in ('CRPD_OUTPUT_TAG','CRPD_CACHE_TAG','CRPD_GPU','CRPD_COMPACT'):
            monkeypatch.delenv(key,raising=False)
        for key,value in {'CRPD_SCALE':'x10','CRPD_TEACHERS':'hda',**env}.items():
            monkeypatch.setenv(key,value)
        spec = importlib.util.spec_from_file_location('qwen3_pilot_cache',Path('script/modal/qwen3_pilot.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    assert (load().GPU_TAG,load().CACHE_SUFFIX) == ('','')
    old = load(CRPD_OUTPUT_TAG='gamma')
    assert (old.GPU_TAG,old.CACHE_SUFFIX) == ('-gamma','-gamma')
    new = load(CRPD_OUTPUT_TAG='g01',CRPD_CACHE_TAG='shared')
    assert (new.GPU_TAG,new.CACHE_SUFFIX) == ('-g01','-shared')
    compact = load(CRPD_OUTPUT_TAG='g01',CRPD_CACHE_TAG='shared',CRPD_COMPACT='1')
    assert (compact.GPU_TAG,compact.CACHE_SUFFIX) == ('-g01-compact','-shared-compact')
    assert "'CRPD_CACHE_TAG':CACHE_TAG" in Path('script/modal/qwen3_pilot.py').read_text(encoding='utf-8')


def test_x30_pilot_matches_x10_splits(monkeypatch) -> None:
    """x30 (2,000-prompt pilot) keeps x10's UltraFeedback rows, states and speed settings, so its select and test
    prompts equal x10's and its train prompts contain x10's; own app and volume (DECISIONS 90)."""
    def load(scale):
        """Import the launcher at one scale."""
        for key,value in {'CRPD_SCALE':scale,'CRPD_TEACHERS':'hda','CRPD_TURBO':'1'}.items():
            monkeypatch.setenv(key,value)
        spec = importlib.util.spec_from_file_location('qwen3_pilot_x30',Path('script/modal/qwen3_pilot.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    x10,x30 = load('x10'),load('x30')
    assert x30.APP_NAME == 'crpd-qwen3-pilot-hda-x30'
    assert x30.SIZES['subset'] == {'train':2000,'head_val':200,'select':64}
    assert (x30.SIZES['steps'],x30.SIZES['selection_interval']) == (450,150)
    for key in ('uf_train_rows','uf_test_rows','states_per_prompt','max_prompt_tokens','gpu_sampling',
                'policy_tokens_per_microbatch','shared_prefix_scoring'):
        assert x30.SIZES[key] == x10.SIZES[key]
