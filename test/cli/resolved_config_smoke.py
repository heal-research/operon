#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real CLI configuration contract tests, including conformance of every printed document to
docs/schemas/resolved-config.schema.json (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import schema_lite


def run(binary, args, ok=True):
    result = subprocess.run([binary, *args], text=True, capture_output=True, timeout=30)
    assert result.returncode == (0 if ok else 1), (args, result.returncode, result.stderr)
    if not ok:
        assert 'error:' in result.stderr, result.stderr
        assert not result.stdout, result.stdout
    return result


def document(text):
    """Parse a document the CLI produced and require it to conform to docs/schemas/resolved-config.schema.json."""
    doc = json.loads(text)
    errors = schema_lite.validate(doc, 'resolved-config.schema.json')
    assert not errors, errors[:5]
    return doc


def normalized(doc):
    doc = json.loads(json.dumps(doc))
    doc['engine'] = {'program': doc['engine']['program']}
    doc['argv'] = []
    doc['data']['file'] = '<dataset>'
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gp', required=True)
    parser.add_argument('--nsgp', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(args.out or tmp)
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / 'data.csv'
        dataset.write_text('x,y\n' + ''.join(f'{i},{i*i+1}\n' for i in range(12)))
        base = ['--dataset', str(dataset), '--target', 'y', '--train', '0:8', '--test', '8:12',
                '--seed', '7', '--threads', '1', '--population-size', '8', '--generations', '1',
                '--objective', 'nmse', '--creator', 'ptc2:0.5:2:4:8', '--maxlength', '8', '--maxdepth', '4',
                '--crossover-internal-probability', '0.5', '--mutators', 'onepoint:2,changevar']
        for algorithm, binary in [('gp', args.gp), ('nsga2', args.nsgp)]:
            printed = run(binary, [*base, '--print-config'])
            doc = document(printed.stdout)
            (root / f'{algorithm}.json').write_text(printed.stdout)
            assert doc['algorithm'] == algorithm
            assert doc['data']['file_sha256'] == hashlib.sha256(dataset.read_bytes()).hexdigest()
            assert doc['operators']['reinserter'] == {'name': 'keep-best', 'elite_count': int(algorithm == 'gp')}
            assert doc['pool_size'] == 1000
            assert doc['budgets']['evaluations']['limit'] == 1000000
            paths = ['operators.reinserter.name', 'operators.reinserter.elite_count', 'pool_size', 'budgets.evaluations.limit']
            assert set(paths) <= set(doc['defaulted'])
            assert doc['operators']['mutation']['mutators'] == [{'name': 'onepoint', 'weight': 2}, {'name': 'changevar', 'weight': 1}]
            caps = json.loads(run(binary, ['--capabilities']).stdout)
            for key in ('program', 'revision', 'precision', 'build_type', 'capabilities'):
                assert doc['engine'][key] == caps[key]
            golden = Path(__file__).with_name(f'resolved_config_{algorithm}.json')
            assert normalized(doc) == json.loads(golden.read_text()), (algorithm, normalized(doc))
            explicit = [*base, '--pool-size', '8', '--reinserter', 'replace-worst', '--elitism', '2', '--evaluations', 'unlimited']
            out = root / f'{algorithm}-both.json'
            both = run(binary, [*explicit, '--config-out', str(out), '--print-config'])
            assert both.stdout == out.read_text()
            actual = document(both.stdout)
            assert not set(paths) & set(actual['defaulted'])
            assert actual['budgets']['evaluations']['limit'] is None
            assert actual['operators']['reinserter'] == {'name': 'replace-worst', 'elite_count': 2}
            real = root / f'{algorithm}-run.json'
            executed = run(binary, [*explicit, '--config-out', str(real)])
            assert 'iteration' in executed.stdout
            assert normalized(document(real.read_text())) == normalized(actual)
            assert not list(root.glob('*.tmp*'))
            for extra in [['--threads', '0'], ['--reinserter', 'invalid'], ['--population-size', '0'],
                          ['--evaluations', 'bad'], ['--objective', 'not-a-metric'], ['--offspring-generator', 'os:bad']]:
                # Replace rather than duplicate existing options.
                override = base[:]
                if extra[0] in override:
                    idx = override.index(extra[0]); del override[idx:idx+2]
                run(binary, [*override, *extra, '--print-config'], ok=False)
            missing = root / 'missing' / 'out.json'
            run(binary, [*explicit, '--config-out', str(missing)], ok=False)
            unlimited = base[:]
            idx = unlimited.index('--generations'); unlimited[idx+1] = 'unlimited'
            unlimited_doc = document(run(binary, [*unlimited, '--print-config']).stdout)
            assert unlimited_doc['budgets']['generations'] is None
            assert unlimited_doc['budgets']['time_limit_s'] is None
            for spec, field, value in [('os:17:0.5', 'max_selection_pressure', 17), ('brood:7', 'size', 7), ('poly:3', 'size', 3)]:
                gen = document(run(binary, [*base, '--offspring-generator', spec, '--print-config']).stdout)
                assert gen['operators']['generator'][field] == value
            shape_path = root / 'shape.json.txt'
            shape_path.write_text(json.dumps({'domains': {'x': [0, 11]}, 'constraints': [{'op': 'id', 'sign': 1}]}))
            shape_args = [*base, '--shape-constraints-config', str(shape_path), '--shape-bound-mode', 'interval:2']
            if algorithm == 'nsga2':
                shape_args += ['--shape-enforcement', 'extra-objective']
            shaped = run(binary, [*shape_args, '--print-config'])
            shaped_doc = document(shaped.stdout)
            names = [obj['name'] for obj in shaped_doc['objective']['objectives']]
            assert names == (['nmse'] if algorithm == 'gp' else ['nmse', 'length', 'violation'])
            assert shaped_doc['shape']['config_sha256'] == hashlib.sha256(shape_path.read_bytes()).hexdigest()
            assert shaped_doc['shape']['bound'] == {'backend': 'interval', 'depth': 2}
            (root / f'{algorithm}-shape.json').write_text(shaped.stdout)
            implicit = base[:]
            for option in ['--seed', '--threads']:
                idx = implicit.index(option); del implicit[idx:idx+2]
            implicit_doc = document(run(binary, [*implicit, '--print-config']).stdout)
            assert implicit_doc['seed_source'] == 'random_device'
            assert implicit_doc['threads']['source'] == 'hardware_concurrency'
            assert implicit_doc['threads']['effective'] >= 1
        print('resolved_config: PASS (both programs; golden documents, defaults, atomic output, invalid options, unlimited budgets, generators, shape objectives, provenance)')


if __name__ == '__main__':
    main()
