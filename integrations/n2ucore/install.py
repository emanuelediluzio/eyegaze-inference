"""Install the gaze overlay into an existing N2uCore checkout (Python 3.10+)."""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IRIDE = 'N2uCore/res/web/iride/'
MERGED = ['N2uCore/res/dyndefs.json', IRIDE + 'schema-ui.json']


def edit_once(text, before, after, filename):
    if after in text:
        return text
    if text.count(before) != 1:
        raise ValueError(f'{filename}: incompatible framework; review the integration manually')
    return text.replace(before, after, 1)


def plan(target):
    if not (target / 'N2uCore/dyn/wrk/mldevice.py').is_file():
        raise ValueError('Target must be an existing N2uCore checkout with MlDevice/MlTicker')
    changes = {}
    for folder in ['N2uCore', 'examples', 'tests', 'tools']:
        for source in sorted((ROOT / folder).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts:
                continue
            rel = source.relative_to(ROOT).as_posix()
            if rel in MERGED:
                current = json.loads((target / rel).read_text(encoding='utf-8'))
                extra = json.loads(source.read_text(encoding='utf-8'))
                if 'nodes' in extra:
                    current['nodes'].update(extra['nodes'])
                else:
                    current.update(extra)
                changes[rel] = (json.dumps(current, indent=2, ensure_ascii=False) + '\n').encode()
            else:
                changes[rel] = source.read_bytes()
    edits = {
        IRIDE + 'lib/model/node-schema.js': [
            ('"fixedLength", "fixedlength", "step", "itemType", "exts"',
             '"fixedLength", "fixedlength", "step", "itemType", "exts", "optional"')],
        IRIDE + 'lib/model/schema-ui.js': [
            ('static UI_KEYS = ["label", "visibility"];',
             'static UI_KEYS = ["label", "visibility", "optional"];')],
        IRIDE + 'lib/model/pipeline-check.js': [
            ('if (typeof spec.type === "string") {\n        const blank',
             'if (typeof spec.type === "string") {\n        if (spec.optional === true) continue; // Explicitly optional text can be empty.\n        const blank')],
        IRIDE + 'lib/ui/forms/node-panels.js': [
            ('import { SwarmPanel } from "./swarm-panel.js";',
             'import { ScreenGazePanel } from "./screen-gaze-panel.js";\nimport { SwarmPanel } from "./swarm-panel.js";'),
            ('static PANELS = [',
             'static PANELS = [\n    ["screenpointofgaze.screenpointofgaze", ScreenGazePanel],')],
    }
    hook_path = ROOT / 'framework-hooks.json'
    if hook_path.is_file():
        for rel, replacements in json.loads(hook_path.read_text(encoding='utf-8')).items():
            edits.setdefault(rel, []).extend(replacements)
    for rel, replacements in edits.items():
        text = (target / rel).read_text(encoding='utf-8')
        for before, after in replacements:
            # Imports/registrations may already be present alongside other panels.
            if rel.endswith('node-panels.js') and after.splitlines()[-1 if before.startswith('static') else 0].strip() in text:
                continue
            text = edit_once(text, before, after, rel)
        changes[rel] = text.encode()
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
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Preflight failed; no changes written: {exc}\n')
    for rel in changes:
        print(rel)
    if not args.apply:
        print(f'Preview: {len(changes)} files. Run with --apply to install.')
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
    print(f'Installed {len(changes)} files. Existing files backed up under {backup}')
    print('Review the diff, provision the checkpoint/dependencies, then restart services when idle.')


if __name__ == '__main__':
    main()
