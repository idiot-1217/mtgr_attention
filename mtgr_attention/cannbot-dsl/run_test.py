"""基于竞品run_test.py增量适配ACLNN；原精度函数、汇总分支和msprof流程复用。

原始来源：tilelang_mtgr_attention/mtgr_attention/tilelang-ascend/run_test.py。
移除FIA/Mask第三方对比依赖；增补锁外参考、真实ACLNN批次及泛化目录入口。
"""
import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from testcase import test_configs, prepare_data
from golden import golden_attention_float64, golden_attention_simulated_kernel
from tilelang_precision_checker import PrecisionLevel, dual_benchmark_precision_check, bootstrap_retest

KERNEL_NAMES = {'sparse-fa': 'kernel_MtgrRaggedSegmentAttention___call___0'}
TOTAL_WARMUP = 5
TOTAL_REPEAT = 10
KERNEL_WARMUP = 5
KERNEL_REPEAT = 5
PERF_INDICES = (0, 3, 26, 72, 96, 147, 442)
A3_US = (228.86, 2687.67, 1420.08, 187.67, 2577.04, 2740.46, 226.55)


def run_kernel(data, config):
    """增量：原接口顺序不变，经真实ACLNN两段式执行，metadata由原生入口生成。"""
    from aclnn_backend import mtgr_ragged_segment_attention
    dev = data.get('_device_data')
    if dev is None:
        dev = {key: value.to('npu:0') if isinstance(value, torch.Tensor) else value
               for key, value in data.items() if not key.startswith('_')}
        data['_device_data'] = dev
    prefix_lens = config['matched_prefix_arr']
    match_mode = 0 if not any(prefix_lens) else 1 if all(prefix_lens) else 2
    output_snd = data.get('_output')
    if output_snd is None:
        output_snd = torch.full_like(dev['query_snd'], float('nan'))
    attributes = dict(block_M=config.get('block_M', 128), core_num=config.get('core_num', 32),
                      num_stages=config.get('num_stages', 14), kv_group=data['kv_group'],
                      cross_interval=config.get('cross_interval', 2), rule1_seg_idx=config.get('rule1_seg_idx', 1))
    attributes.update(config.get('_attributes', {}))
    return mtgr_ragged_segment_attention(
        dev['query_snd'], dev['key_snd'], dev['value_snd'], dev['segment_offsets_i32'],
        dev['segment_rules_i32'], dev['q_seq_starts_i32'], dev['matched_prefix_lens_i32'], match_mode,
        dev['key_cache'], dev['value_cache'], dev['block_table_tensor'], dev['block_size'],
        max(map(sum, config['seg_lengths'])), data['sm_scale'], output_snd, metadata=None, **attributes)


def _rle(values):
    if not values:
        return []
    result = []
    cur = values[0]
    count = 1
    for v in values[1:]:
        if v == cur:
            count += 1
        else:
            result.append(f"{cur}x{count}" if count >= 2 else str(cur))
            cur = v
            count = 1
    result.append(f"{cur}x{count}" if count >= 2 else str(cur))
    return result

def _serialize_seg_lengths(seg_lengths):
    parts = []
    for seg in seg_lengths:
        if len(seg) <= 2:
            parts.append("|".join(str(x) for x in seg))
        else:
            tokens = [str(seg[0])] + _rle(seg[1:-1]) + [str(seg[-1])]
            parts.append("|".join(tokens))
    return ";".join(parts)

def _serialize_list(lst):
    return "|".join(_rle(lst))

def _case_row(case_index, config):
    seg_lengths = config["seg_lengths"]
    return {
        "case_index": case_index,
        "H": config["H"],
        "D": config["D"],
        "B": len(seg_lengths),
        "num_segs": len(seg_lengths[0]),
        "seg_lengths": _serialize_seg_lengths(seg_lengths),
        "rules": _serialize_list(config["rules"]),
        "matched_prefix_arr": _serialize_list(config["matched_prefix_arr"]),
    }

def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  results written to {path}")

def _golden_args(data):
    return (
        data["query_snd"],
        data["key_snd"],
        data["value_snd"],
        data["segment_offsets_i32"],
        data["segment_rules_i32"],
        data["q_seq_starts_i32"],
        data["matched_prefix_lens_i32"],
        data["key_cache"],
        data["value_cache"],
        data["block_table_tensor"],
        data["block_size"],
        data["sm_scale"],
    )

def run_perf_timed(data, config, warmup, repeat):
    for _ in range(warmup):
        run_kernel(data, config)
        torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        run_kernel(data, config)
        torch.npu.synchronize()
    t1 = time.perf_counter()

    total_us = (t1 - t0) * 1_000_000
    avg_us = total_us / repeat
    print(f"  total: {total_us:.2f}us  avg: {avg_us:.2f}us  ({repeat} runs)")
    return total_us

def case_label(idx, config):
    name = config.get("name", "")
    B = len(config["seg_lengths"])
    D = config["D"]
    if name:
        return f"Case[{idx}] {name}"
    return f"Case[{idx}] D{D}_batch{B}"

def parse_kernel_times(log_dir):
    pattern = os.path.join(log_dir, "OPPROF_*", "**", "OpBasicInfo*.csv")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        pattern = os.path.join(log_dir, "OPPROF_*", "OpBasicInfo.csv")
        matches = glob.glob(pattern)
    if not matches:
        return None
    entries = []
    for csv_path in matches:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Op Name", "").strip()
                dur = row.get("Task Duration(us)", "").strip()
                if name and dur:
                    entries.append({"name": name, "duration_us": float(dur)})
    if not entries:
        return None
    return entries

_ACC_FIELDS = [
    "case_index",
    "H",
    "D",
    "B",
    "num_segs",
    "seg_lengths",
    "rules",
    "matched_prefix_arr",
    "total_runs",
    "passed",
    "status",
    "mare_min",
    "mare_max",
    "mere_min",
    "mere_max",
    "rmse_min",
    "rmse_max",
]

def run_accuracy(data, config, golden_sim, golden_f64, verbose=False, native_outputs=None):
    if native_outputs is None:
        _ = run_kernel(data, config)
        torch.npu.synchronize()
        output_snd = run_kernel(data, config)
        torch.npu.synchronize()
    else:
        # 增量：仅接收本轮设备worker经两次真实ACLNN执行保存的输出，在锁外调用原检查器。
        _, output_snd = native_outputs

    report = dual_benchmark_precision_check(
        npu_output=output_snd.cpu(),
        ref_output=golden_sim.cpu(),
        golden=golden_f64.cpu(),
        level=PrecisionLevel.L0,
        verbose=verbose,
    )
    return report.passed, report

def aggregate_accuracy(reports, extra_hard_fail=False, no_bootstrap=False):
    """从原main抽取汇总分支，判定及Bootstrap参数保持原样。"""
    case_total = len(reports)
    case_pass = sum(report.passed for report in reports)
    hard_fail = extra_hard_fail or any(not report.inf_nan_match or not report.small_value_passed for report in reports)
    mare_ratio_list = [r.mare_ratio for r in reports]
    mere_ratio_list = [r.mere_ratio for r in reports]
    rmse_ratio_list = [r.rmse_ratio for r in reports]
    args = argparse.Namespace(no_bootstrap=no_bootstrap)
    if case_pass == case_total:
        case_passed = True
    elif hard_fail:
        case_passed = False
        print("  hard fail: INF/NAN or SmallValue failed, skip bootstrap")
    elif not args.no_bootstrap:
        bs = bootstrap_retest(
            mare_ratio_list,
            mere_ratio_list,
            rmse_ratio_list,
            verbose=True,
        )
        case_passed = bs["passed"]
    else:
        case_passed = False
    return case_passed



def run_msprof_case(case_index, config, log_dir, target='sparse-fa', warmup=KERNEL_WARMUP, repeat=KERNEL_REPEAT):
    """基于原msprof函数增量适配：采集10次主kernel，明确排除应用前5次预热。"""
    from validation_support import run_profile_command, write_report
    target_log_dir = Path(log_dir) / f'case{case_index}'
    target_log_dir.mkdir(parents=True, exist_ok=False)
    data = prepare_data(config, 0)
    torch.save(dict(data=data, config=config), target_log_dir / 'payload.pt')
    cmd = ['msprof', 'op', f'--kernel-name=*{KERNEL_NAMES[target]}*',
           f'--output={target_log_dir / "profile"}', '--aic-metrics=BasicInfo',
           f'--warm-up={warmup}', f'--launch-count={warmup + repeat}', '--kill=true',
           sys.executable, str(Path(__file__).resolve()), '--_msprof-worker',
           '--case-index', str(case_index), '--_msprof-repeat', str(warmup + repeat),
           '--_msprof-output', str(target_log_dir), '--output-dir', str(target_log_dir)]
    run_profile_command(cmd, target_log_dir)
    files = list((target_log_dir / 'profile').rglob('OpBasicInfo*.csv'))
    assert len(files) == warmup + repeat, f'Expected 10 main launches, got {len(files)}.'
    files.sort(key=lambda path: int(path.parent.name))
    assert [int(path.parent.name) for path in files] == list(range(warmup + repeat))
    entries = parse_kernel_times(str(target_log_dir / 'profile'))
    assert entries and len(entries) == warmup + repeat
    measured = []
    measured_records = []
    for iteration, path in enumerate(files):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and rows[0]['Op Name'] == KERNEL_NAMES[target]
        assert rows[0]['Device Id'] == '0' and rows[0]['Block Dim'] == '32' and rows[0]['Mix Block Dim'] == '64'
        if iteration >= warmup:
            measured.append(float(rows[0]['Task Duration(us)']))
            measured_records.append(dict(iteration=iteration, csv_path=str(path), **rows[0]))
    assert len(measured) == repeat and all(math.isfinite(x) and x > 0 for x in measured)
    baseline = A3_US[PERF_INDICES.index(case_index)]
    mean = sum(measured) / repeat
    result = dict(case_index=case_index, samples_us=measured, kernel_avg_us=mean,
                  A3_us=baseline, target_us=baseline / 3, speedup=baseline / mean,
                  passed=mean <= baseline / 3, warmup=warmup, repeat=repeat,
                  profiler_warmup=warmup, collected_main_launches=warmup + repeat,
                  selected_application_iterations=list(range(warmup, warmup + repeat)), command=cmd)
    result['raw_measured_records'] = measured_records
    write_report(target_log_dir / 'performance.txt', result)
    return result


def main():
    from validation_support import accuracy_suite, accuracy_worker, worker_initialize, write_report, signature, ROOT
    parser = argparse.ArgumentParser(description='Original sparse attention test runner: ACLNN incremental adaptation')
    parser.add_argument('--mode', choices=['accuracy', 'compare', 'acceptance'], default='accuracy')
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--case-indices')
    parser.add_argument('--acc-seeds', type=int, default=3)
    parser.add_argument('--acc-repeat', type=int, default=1)
    parser.add_argument('--acc-verbose-fail', action='store_true')
    parser.add_argument('--no-bootstrap', action='store_true')
    parser.add_argument('--all-multi-seed', action='store_true')
    parser.add_argument('--all-perf', action='store_true')
    parser.add_argument('--msprof-log')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-cache', action='append', default=[])
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--generalization', type=int, default=0)
    parser.add_argument('--acceptance-report')
    parser.add_argument('--_accuracy-worker', type=Path)
    parser.add_argument('--_msprof-worker', action='store_true')
    parser.add_argument('--_msprof-repeat', type=int, default=10)
    parser.add_argument('--_msprof-output', type=Path)
    args = parser.parse_args()
    if args._accuracy_worker:
        accuracy_worker(args._accuracy_worker)
        return
    if args._msprof_worker:
        payload = torch.load(args._msprof_output / 'payload.pt', weights_only=True)
        worker_initialize(False)
        data, config = payload['data'], payload['config']
        data['_device_data'] = {key: value.to('npu:0') if isinstance(value, torch.Tensor) else value for key,value in data.items()}
        data['_output'] = torch.empty_like(data['_device_data']['query_snd'])
        for iteration in range(args._msprof_repeat):
            run_kernel(data, config)
            torch.npu.synchronize()
            print('PROFILE_APPLICATION_ITERATION', iteration, flush=True)
        return
    assert 1 <= args.workers <= 72 and 1 <= args.batch_size <= 128
    assert args.acc_seeds >= 3 and 0 <= args.generalization <= 10000
    if args.generalization:
        assert args.acceptance_report and args.mode == 'accuracy'
    output = Path(args.output_dir).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    frozen = signature()
    accuracy = accuracy_suite(args) if args.mode in ('accuracy', 'acceptance') else None
    if accuracy and not accuracy['passed']:
        raise SystemExit(1)
    performance = []
    if args.mode in ('compare', 'acceptance'):
        from aclnn_backend import precompile
        precompile({}, {}, core_num=32)
        indices = [args.case_index] if args.case_index is not None else [int(x) for x in args.case_indices.split(',')] if args.case_indices else list(PERF_INDICES)
        for index in indices:
            assert index in PERF_INDICES, 'A5 performance acceptance contains exactly seven named cases.'
            performance.append(run_msprof_case(index, test_configs[index], args.msprof_log or str(output / 'performance')))
            assert signature() == frozen
    import importlib.metadata
    environment = dict(torch=importlib.metadata.version('torch'), torch_npu=importlib.metadata.version('torch-npu'),
                       cann=os.environ.get('ASCEND_HOME_PATH', ''), device=0, cube_cores=32, npu_subtask_limit_seconds=300)
    environment['hardware'] = subprocess.check_output(['asys','info','-r=status','-d','0'], text=True, timeout=10)
    report = dict(environment=environment, complete=True, passed=(not accuracy or accuracy['passed']) and all(x['passed'] for x in performance),
                  accuracy=accuracy, performance=performance, source_sha256=frozen,
                  method='Original golden.py/testcase.py/run_test.py precision logic; real ACLNN; msprof 5 warmups + mean of 5')
    write_report(output / 'validation_report.txt', report)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
