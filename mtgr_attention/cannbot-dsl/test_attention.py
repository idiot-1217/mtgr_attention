"""通过pytest调用增量适配后的竞品run_test.py，不另写精度或性能判定。"""
import subprocess
import sys
from pathlib import Path
import pytest


def test_attention(request):
    if not request.config.getoption('--mtgr-npu'):
        pytest.skip('Use --mtgr-npu in the configured Ascend environment.')
    output = request.config.getoption('--mtgr-output')
    assert output, 'An explicit output directory is required.'
    args = [sys.executable, str(Path(__file__).with_name('run_test.py')), '--mode', request.config.getoption('--mtgr-mode'),
            '--output-dir', output, '--workers', str(request.config.getoption('--mtgr-workers'))]
    for key, option in [('--mtgr-cases','--case-indices'),('--mtgr-acceptance-report','--acceptance-report')]:
        if request.config.getoption(key):args.extend([option, request.config.getoption(key)])
    for cache in request.config.getoption('--mtgr-reference-cache'):args.extend(['--reference-cache',cache])
    if request.config.getoption('--mtgr-generalization'):args.extend(['--generalization',str(request.config.getoption('--mtgr-generalization'))])
    subprocess.run(args,check=True)
