"""Install the gaze overlay into an existing N2uCore checkout (Python 3.10+)."""
import argparse
import ast
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IRIDE = 'N2uCore/res/web/iride/'
MERGED = ['N2uCore/res/dyndefs.json', 'N2uCore/res/uidefs.json', IRIDE + 'schema-ui.json']
# Iride hooks the gaze panels need; they are part of the framework Iride (N2uCore_dev), not of this package
IRIDE_HOOKS = {
    IRIDE + 'app/run/card.js': 'panels/node-panels.js',
    IRIDE + 'app/run/output-cards.js': 'panels/node-panels.js',
    IRIDE + 'app/run/card-board.js': 'panels/panel-sessions.js',
    IRIDE + 'index.html': 'gaze-panels.css',
    IRIDE + 'lib/render/data-worker.js': 'gaze-decoder.js',
    IRIDE + 'lib/render/renderers/overlay-renderer.js': 'gaze-direction-drawing.js',
    IRIDE + 'app/texts/en.js': '"gaze.calibrate"',
    IRIDE + 'app/texts/it.js': '"gaze.calibrate"',
}


def plan(target):
    if not (target / 'N2uCore/dyn/wrk/mldevice.py').is_file():
        raise ValueError('Target must be an existing N2uCore checkout with MlDevice/MlTicker')
    provider = target / 'N2uCore/dyn/wrk/memprovider.py'
    if not provider.is_file():
        raise ValueError('Update the framework ML resource API before installing Eye Gaze')
    functions = {node.name for node in ast.parse(provider.read_text(encoding='utf-8')).body
                 if isinstance(node, ast.FunctionDef)}
    if not {'resolveComputeResources', 'releaseComputeCache'} <= functions:
        raise ValueError('Framework memprovider.py must provide resolveComputeResources and releaseComputeCache; framework ML changes are distributed separately')
    for rel, hook in IRIDE_HOOKS.items():
        if not (target / rel).is_file() or hook not in (target / rel).read_text(encoding='utf-8'):
            raise ValueError(f'{rel}: no gaze panel hook ({hook}); the target Iride must be the N2uCore_dev one')
    changes = {}
    for folder in ['N2uCore', 'examples', 'tests', 'tools', 'docs']:
        for source in sorted((ROOT / folder).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts:
                continue
            rel = source.relative_to(ROOT).as_posix()
            if rel == 'N2uCore/dyn/wrk/memprovider.py' or rel.startswith('tests/dyn/wrk/'):
                continue
            if rel in MERGED:
                current = json.loads((target / rel).read_text(encoding='utf-8'))
                extra = json.loads(source.read_text(encoding='utf-8'))
                if all(current.get(key) == value for key, value in extra.items()):
                    continue
                current.update(extra)
                changes[rel] = (json.dumps(current, indent=2, ensure_ascii=False) + '\n').encode()
            else:
                changes[rel] = source.read_bytes()
    return {rel: data for rel, data in changes.items()
            if not (target / rel).is_file() or (target / rel).read_bytes() != data}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path, help='Existing N2uCore repository directory')
    parser.add_argument('--apply', action='store_true', help='Write changes after complete preflight; default is preview')
    args = parser.parse_args()
    target = args.target.expanduser().resolve()
    try:
        changes = plan(target)
        legacy = target / 'N2uCore/inference/gazedino.py'
        removals = []
        if legacy.is_file():
            source = legacy.read_text(encoding='utf-8')
            if 'class GazeDinoEngine' not in source or 'class GazeDINO' not in source:
                raise ValueError('Legacy gazedino.py differs from the gaze integration; migrate it manually')
            removals.append(legacy.relative_to(target).as_posix())
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Preflight failed; no changes written: {exc}\n')
    for rel in changes:
        print(rel)
    for rel in removals:
        print(f'Remove legacy engine: {rel}')
    if not args.apply:
        print(f'Preview: {len(changes)} files, {len(removals)} legacy removals. Run with --apply to install.')
        return
    backup = target / '.gaze-integration-backups' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    for rel, data in changes.items():
        out = target / rel
        if out.exists():
            saved = backup / rel
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, saved)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    for rel in removals:
        out = target / rel
        saved = backup / rel
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, saved)
        out.unlink()
    print(f'Installed {len(changes)} files. Existing files backed up under {backup}')
    print('Review the diff, provision the checkpoint/dependencies, then restart services when idle.')


if __name__ == '__main__':
    main()
