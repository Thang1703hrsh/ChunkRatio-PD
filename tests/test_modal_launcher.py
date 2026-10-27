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
