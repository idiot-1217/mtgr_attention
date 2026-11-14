"""竞品测试脚本的增量运行支持：锁外原始参考、真实ACLNN批次和冻结报告。"""
import ast
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import dataclasses
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE
sys.path.insert(0, str(SOURCE))
ROOT = Path(os.environ.get('MTGR_VALIDATION_ROOT', str(HERE))).resolve()
LOCK = Path('/tmp/cannbotdsl-validation-device0.lock')


def write_report(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def holders(ignore_group=None):
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if ignore_group is not None and os.getpgid(int(proc.name)) == ignore_group:
                continue
            for fd in (proc / 'fd').iterdir():
                try:
                    if os.readlink(fd) == '/dev/davinci0':
                        found.append(int(proc.name))
                        break
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


def signature():
    paths = list(HERE.glob('*.py')) + [p for p in SOURCE.iterdir() if p.suffix in ('.py', '.cpp', '.h')]
    result = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    library = Path(os.environ['MTGR_ACLNN_LIBRARY'])
    result['aclnn_library'] = hashlib.sha256(library.read_bytes()).hexdigest()
    exported = library.parent.parent
    for p in sorted(exported.rglob('*')):
        if p.is_file() and (p.suffix in ('.so', '.o', '.json') or '/kernel/' in str(p)):
            result['native/' + str(p.relative_to(exported))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def initialize_cpu():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def prepare_task(arguments):
    import torch
    import run_test as runner
    task, folder, cache_roots = arguments
    config = task['config']
    data = runner.prepare_data(config, task['seed'])
    entry = task.get('entry')
    if entry:
        data['sm_scale'] = entry['scale']
        for key in {'zero_q': ('query_snd',), 'zero_k': ('key_snd', 'key_cache'),
                    'zero_v': ('value_snd', 'value_cache')}.get(entry['pattern'], ()):
            data[key].zero_()
        if entry['pattern'] == 'constant_v':
            data['value_snd'].fill_(0.25)
            data['value_cache'].fill_(0.25)
        if entry['shuffle_pages']:
            permutation = torch.randperm(data['key_cache'].shape[0])
            inverse = torch.argsort(permutation)
            data['key_cache'] = data['key_cache'][permutation].contiguous()
            data['value_cache'] = data['value_cache'][permutation].contiguous()
            data['block_table_tensor'] = inverse[data['block_table_tensor'].long()].to(torch.int32)
    if not entry:
        for root in cache_roots:
            candidate = Path(root) / f"case{task['case_index']}/seed{task['seed']}/payload.pt"
            if not candidate.exists():
                continue
            packed = torch.load(candidate, weights_only=True)
            current, previous = runner._golden_args(data), runner._golden_args(packed['data'])
            if packed['config'] == config and packed['seed'] == task['seed'] and all(
                torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b for a, b in zip(current, previous)
            ):
                return dict(task=task, payload=str(candidate), reference_cache=True)
    high = runner.golden_attention_float64(*runner._golden_args(data))
    simulated = runner.golden_attention_simulated_kernel(*runner._golden_args(data))
    path = Path(folder) / f"case{task['case_index']}_seed{task['seed']}.pt"
    temporary = path.with_suffix('.pending')
    torch.save(dict(data=data, high=high, simulated=simulated), temporary)
    temporary.replace(path)
    return dict(task=task, payload=str(path), reference_cache=False)


def worker_initialize(handshake):
    import torch
    from aclnn_backend import libraries, precompile
    precompile({}, {}, core_num=32)
    libraries()
    for fd in (Path('/proc/self/fd').iterdir() if handshake else []):
        try:
            assert os.readlink(fd) != '/dev/davinci0'
        except OSError:
            pass
    if handshake:
        os.write(int(os.environ['MTGR_PREPARE_READY_FD']), b'1')
        os.close(int(os.environ['MTGR_PREPARE_READY_FD']))
        assert os.read(int(os.environ['MTGR_DEVICE_GRANT_FD']), 1) == b'1'
        os.close(int(os.environ['MTGR_DEVICE_GRANT_FD']))
    assert os.readlink('/proc/self/fd/' + os.environ['MTGR_NPU_LOCK_FD']) == str(LOCK)
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '0' and not holders(ignore_group=os.getpgrp())
    import torch_npu  # noqa: F401
    torch.npu.set_device('npu:0')


def accuracy_worker(folder):
    import torch
    import run_test as runner
    items = json.loads((folder / 'items.txt').read_text())
    loaded = []
    for item in items:
        data = torch.load(item['payload'], weights_only=True)['data']
        loaded.append((item, data))
    worker_initialize(True)
    for ordinal, (item, data) in enumerate(loaded):
        config = dict(item['task']['config'])
        if item['task'].get('entry'):
            config['_attributes'] = item['task']['entry']['attributes']
        first = runner.run_kernel(data, config).cpu()
        actual = runner.run_kernel(data, config).cpu()
        torch.save(dict(first=first, actual=actual), folder / f'actual{ordinal}.pt')
        print('ACLNN_EXECUTED', item['task']['case_index'], item['task']['seed'], flush=True)


def evaluate_item(arguments):
    import torch
    import run_test as runner
    item, output = arguments
    packed = torch.load(item['payload'], weights_only=True)
    native = torch.load(output, weights_only=True)
    first, actual = native['first'], native['actual']
    finite = all(bool(torch.isfinite(x).all()) for x in (actual, first, packed['high'], packed['simulated']))
    deterministic = bool(torch.equal(first.view(torch.uint16), actual.view(torch.uint16)))
    ok, report = runner.run_accuracy(packed['data'], item['task']['config'], packed['simulated'], packed['high'],
                                     native_outputs=(first, actual))
    diagnostics = {}
    for name, reference in [('float64', packed['high']), ('simulated', packed['simulated'])]:
        delta = (actual.double() - reference.double()).abs()
        diagnostics[name] = dict(allclose=bool(torch.allclose(actual.double(), reference.double(), atol=1e-2, rtol=1e-2)),
                                 bad_elements=int((delta > 1e-2 + 1e-2 * reference.double().abs()).sum()))
    zero = packed['high'].abs().sum(dim=-1) == 0
    zero_pass = bool((actual[zero] == 0).all())
    result = dict(item, passed=bool(ok and finite and deterministic and zero_pass),
                  finite=finite, deterministic=deterministic, zero_rows_passed=zero_pass,
                  accuracy=dataclasses.asdict(report), allclose_diagnostic=diagnostics,
                  native_executions=2, actual_path=str(output))
    result['accuracy']['level'] = int(report.level)
    return result


def execute_batch(output, number, items):
    folder = output / f'batch{number:04d}'
    folder.mkdir(exist_ok=True)
    write_report(folder / 'items.txt', items)
    command = [sys.executable, str(HERE / 'bounded_npu.py'), '--seconds', '300', '--',
               sys.executable, str(HERE / 'run_test.py'), '--_accuracy-worker', str(folder), '--output-dir', str(folder)]
    with (folder / 'device.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(8) as checks:
        results = list(checks.map(evaluate_item, [(item, folder / f'actual{i}.pt') for i, item in enumerate(items)]))
    write_report(folder / 'results.txt', results)
    print('BATCH_RESULT', number, len(results), sum(x['passed'] for x in results), flush=True)
    return results


def summarize_accuracy(rows, tasks, no_bootstrap=False):
    import run_test as runner
    grouped = {}
    for task in tasks:
        grouped.setdefault(task['case_index'], []).append(task)
    observed = {}
    for row in rows:
        observed.setdefault(row['task']['case_index'], []).append(row)
    cases = []
    for index, expected in sorted(grouped.items()):
        selected = observed.get(index, [])
        if len(selected) != len(expected):
            continue
        from types import SimpleNamespace
        reports = [SimpleNamespace(**row['accuracy']) for row in selected]
        passed = runner.aggregate_accuracy(reports, extra_hard_fail=any(not row['passed'] for row in selected
                                           if not row['finite'] or not row['deterministic'] or not row['zero_rows_passed']),
                                           no_bootstrap=no_bootstrap)
        # 附加硬检查独立于原脚本汇总分支，任一失败必须让整个用例失败。
        passed = passed and all(row['finite'] and row['deterministic'] and row['zero_rows_passed'] for row in selected)
        cases.append(dict(case_index=index, passed=passed, seeds=[row['task']['seed'] for row in selected]))
    return cases


def accuracy_suite(args):
    import torch
    torch.set_num_threads(1)
    import run_test as runner
    from testcase import make_generalization_catalog
    output = Path(args.output_dir).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    if args.generalization:
        parent = json.loads(Path(args.acceptance_report).read_text())
        assert parent['complete'] and parent['accuracy']['passed'] and len(parent['performance']) == 7
        entries = make_generalization_catalog(args.generalization)
        assert len({e['structural_sha256'] for e in entries}) == args.generalization
        tasks = [dict(case_index=e['case_index'], seed=e['seed'], config=e['config'], entry=e) for e in entries]
    else:
        indices = list(range(len(runner.test_configs))) if not args.case_indices else [int(x) for x in args.case_indices.split(',')]
        if args.case_index is not None:
            indices = [args.case_index]
        assert indices and all(0 <= i < len(runner.test_configs) for i in indices)
        tasks = [dict(case_index=i, seed=seed, config=runner.test_configs[i]) for i in indices
                 for seed in range(args.acc_seeds if runner.test_configs[i].get('multi_seed') or args.all_multi_seed
                                   or i in runner.PERF_INDICES else 1)]
    assert args.acc_repeat == 1, 'Batch protocol uses one measured result after one warmup; set --acc-repeat=1.'
    frozen = signature()
    caches = [Path(x) for x in args.reference_cache]
    for cache in caches:
        manifest = json.loads((cache / 'manifest.txt').read_text())
        old = manifest['source_sha256']
        for current, previous in [('golden.py', 'golden.py'), ('tilelang_precision_checker.py', 'baseline_precision_checker.py')]:
            source = HERE / current
            assert hashlib.sha256(source.read_bytes()).hexdigest() == old.get(previous, old.get('validation/' + previous))
        original_prepare = next(n for n in ast.parse((HERE / 'testcase.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_data')
        prior_prepare = next(n for n in ast.parse((cache.parents[1] / 'snapshot/test/attention/mtgr_ragged_segment_attention/data_generator.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_data')
        assert ast.dump(original_prepare) == ast.dump(prior_prepare)
    identity = dict(source_sha256=frozen, tasks=tasks, reference_cache=[str(x) for x in caches], reference='original golden.py functions',
                    native_output_reuse=False, acceptance='original run_test.run_accuracy and original aggregate branch')
    if (output / 'manifest.txt').exists():
        assert json.loads((output / 'manifest.txt').read_text()) == identity
    else:
        write_report(output / 'manifest.txt', identity)
    records = output / 'seed_results.txt'
    rows = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
    done = {(x['task']['case_index'], x['task']['seed']) for x in rows}
    pending = [task for task in tasks if (task['case_index'], task['seed']) not in done]
    ready = []
    context = multiprocessing.get_context('spawn')
    active = {}
    device = None
    number = max((int(p.name[5:]) for p in output.glob('batch[0-9]*')), default=-1) + 1
    used_memory = 0
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=initialize_cpu) as pool, ThreadPoolExecutor(1) as npu:
        while pending or active or ready or device:
            while pending and len(active) < args.workers:
                task = pending[0]
                c = task['config']
                weight = max(c['H'] * sum(a) * (sum(a) - p) for a,p in zip(c['seg_lengths'], c['matched_prefix_arr'])) * 40 + 1024**3
                if active and used_memory + weight > 600 * 1024**3:
                    break
                pending.pop(0)
                future = pool.submit(prepare_task, (task, str(output), [str(x) for x in caches]))
                active[future] = weight
                used_memory += weight
            for future in list(active):
                if future.done():
                    used_memory -= active.pop(future)
                    ready.append(future.result())
            if device and device.done():
                checked = device.result()
                with records.open('a') as log:
                    for row in checked:
                        log.write(json.dumps(row) + '\n')
                rows.extend(checked)
                device = None
                assert signature() == frozen
                cases = summarize_accuracy(rows, tasks, args.no_bootstrap)
                write_report(output / 'progress.txt', dict(completed_seeds=len(rows), expected_seeds=len(tasks),
                             completed_cases=len(cases), failed_cases=[c['case_index'] for c in cases if not c['passed']]))
                print('ACCURACY_PROGRESS', len(rows), '/', len(tasks), flush=True)
            if not device and ready and (len(ready) >= args.batch_size or not pending and not active):
                batch, ready = ready[:args.batch_size], ready[args.batch_size:]
                device = npu.submit(execute_batch, output, number, batch)
                number += 1
            time.sleep(0.1)
    assert signature() == frozen
    cases = summarize_accuracy(rows, tasks, args.no_bootstrap)
    report = dict(complete=True, passed=len(cases) == len({t['case_index'] for t in tasks}) and all(c['passed'] for c in cases),
                  cases=cases, total_cases=len(cases), total_seeds=len(rows), failed_cases=[c['case_index'] for c in cases if not c['passed']],
                  native_executions=2*len(rows), manifest=identity)
    observed = {}
    for row in rows:
        observed.setdefault(row['task']['case_index'], []).append(row)
    csv_rows = []
    for case in cases:
        selected = observed[case['case_index']]
        csv_rows.append(dict(runner._case_row(case['case_index'], selected[0]['task']['config']),
                        total_runs=len(selected), passed=sum(x['passed'] for x in selected),
                        status='PASS' if case['passed'] else 'FAIL',
                        **{name + suffix: operation(x['accuracy'][metric] for x in selected)
                           for name,metric in [('mare','mare_npu'),('mere','mere_npu'),('rmse','rmse_npu')]
                           for suffix,operation in [('_min',min),('_max',max)]}))
    runner._write_csv(str(output / 'accuracy_results.csv'), csv_rows, runner._ACC_FIELDS)
    report['finite_seeds'] = sum(row['finite'] for row in rows)
    report['deterministic_seeds'] = sum(row['deterministic'] for row in rows)
    report['zero_rows_passed_seeds'] = sum(row['zero_rows_passed'] for row in rows)
    report['reference_cache_seeds'] = sum(row['reference_cache'] for row in rows)
    report['allclose_diagnostic'] = {name: dict(passed_seeds=sum(row['allclose_diagnostic'][name]['allclose'] for row in rows),
                                           bad_elements=sum(row['allclose_diagnostic'][name]['bad_elements'] for row in rows))
                                       for name in ('float64', 'simulated')}
    if args.generalization:
        from collections import Counter
        report['coverage'] = {name: dict(Counter(str(task['config'][name]) for task in tasks)) for name in ('D','kv_group')}
        report['coverage'].update({name: dict(Counter(str(task['entry'][name]) for task in tasks)) for name in ('pattern','match_mode')})
        report['structural_unique_count'] = len({task['entry']['structural_sha256'] for task in tasks})
    write_report(output / 'accuracy_summary.txt', report)
    return report


def run_profile_command(command, folder):
    from aclnn_backend import precompile
    precompile({}, {}, core_num=32)
    write_report(folder / 'idle_check.txt', dict(holders=holders()))
    assert not holders(), 'NPU0 has active processes.'
    write_report(folder / 'command.txt', command)
    wrapped = [sys.executable, str(HERE / 'bounded_npu.py'), '--seconds', '180', '--', *command]
    with (folder / 'msprof.log').open('w') as log:
        subprocess.run(wrapped, stdout=log, stderr=subprocess.STDOUT, check=True)
