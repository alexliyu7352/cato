"""Prevent publishing a broken GPU handoff notebook/fixture bundle."""
import ast
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def test_colab_notebook_source_parses():
    doc = json.loads((ROOT / 'notebooks/R0_Colab_GPU_Validation.ipynb').read_text())
    assert doc['nbformat'] == 4
    codes = [cell for cell in doc['cells'] if cell['cell_type'] == 'code']
    assert len(codes) >= 6
    for cell in codes:
        ast.parse(''.join(cell['source']))
    joined = '\n'.join(''.join(cell['source']) for cell in codes)
    assert 'torch.cuda.is_available()' in joined
    assert "SOURCE_MODE = 'github'" in joined
    assert "GITHUB_REPO = 'alexliyu7352/cato'" in joined
    assert "userdata.get('GH_PAT')" not in joined
    assert "--device', 'cuda'" in joined


def test_fixed_smoke_fixture_archive_contains_manifest():
    fixture = ROOT / 'assets/r0_smoke_fixture.zip'
    with ZipFile(fixture) as z:
        files = set(z.namelist())
    assert 'samples/smoke/manifest.jsonl' in files
    assert sum(f.endswith('/source.png') for f in files) == 12
    assert sum(f.endswith('/depth.npy') for f in files) == 12
