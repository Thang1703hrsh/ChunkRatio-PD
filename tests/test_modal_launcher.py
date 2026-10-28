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
    assert "@app.function(cpu=4.0,memory=16384,**common)\ndef prepare" in source
    assert "@app.function(cpu=0.25,memory=1024,**common)\ndef run_pilot" in source
    assert "gpu='L40S',cpu=4.0,memory=16384,retries=0,**sft_limits)\ndef sft_student" in source
    assert "retries=0,**dpo_limits)\ndef dpo_teacher" in source and module.SFT_DATA.endswith('fit2048.jsonl')
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
