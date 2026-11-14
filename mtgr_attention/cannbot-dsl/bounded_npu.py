"""CPU准备完成后才获取NPU0公共锁；设备任务独立限时并清理进程组。"""
import argparse
import fcntl
import os
from pathlib import Path
import select
import signal
import subprocess
import time


def stop_process(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    # 清理编译器等可能由子进程创建的同组后代。
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 570:
        parser.error("duration must be 1..570s, leaving cleanup time within 600s")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command")
    # 本算子的worker支持先准备、后授权协议；其他命令仍在获取锁后启动。
    prepare_before_lock = (len(command) > 2 and Path(command[1]).resolve() == Path(__file__).with_name('run_test.py').resolve()
                           and '--_accuracy-worker' in command)
    process = None
    lock_started = None
    pipe_fds = []
    with Path("/tmp/cannbotdsl-validation-device0.lock").open("r") as lock:
        env = dict(os.environ, MTGR_NPU_DEVICE="0", MTGR_NPU_LOCK_FD=str(lock.fileno()),
                   ASCEND_RT_VISIBLE_DEVICES="0")
        try:
            if prepare_before_lock:
                ready_read, ready_write = os.pipe()
                grant_read, grant_write = os.pipe()
                pipe_fds.extend((ready_read, ready_write, grant_read, grant_write))
                env.update(MTGR_PREPARE_READY_FD=str(ready_write), MTGR_DEVICE_GRANT_FD=str(grant_read))
                preparation_started = time.monotonic()
                process = subprocess.Popen(command, env=env, start_new_session=True,
                                           pass_fds=(lock.fileno(), ready_write, grant_read))
                for fd in (ready_write, grant_read):
                    os.close(fd)
                    pipe_fds.remove(fd)
                deadline = preparation_started + 240
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("CPU preparation exceeded the 240-second limit.")
                    readable, _, _ = select.select([ready_read], [], [], 1)
                    if readable:
                        if os.read(ready_read, 1) != b"1":
                            raise RuntimeError("The validation worker exited before CPU preparation completed.")
                        break
                    if process.poll() is not None:
                        raise RuntimeError(f"CPU preparation failed with exit code {process.returncode}.")
                print(f"CPU preparation completed outside device lock in {time.monotonic()-preparation_started:.2f}s", flush=True)
            while True:
                if process is not None and process.poll() is not None:
                    raise RuntimeError("The prepared validation worker exited before acquiring the device lock.")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(2)
            lock_started = time.monotonic()
            health = subprocess.run(["asys", "health", "-d", "0"], capture_output=True,
                                    text=True, timeout=10, check=True).stdout
            if "Overall Health: Healthy" not in health or "ErrorCode Num: 0" not in health:
                raise RuntimeError(f"NPU0 is not healthy: {health}")
            if prepare_before_lock:
                os.write(grant_write, b"1")
            else:
                process = subprocess.Popen(command, env=env, start_new_session=True,
                                           pass_fds=(lock.fileno(),))
            try:
                result = process.wait(timeout=args.seconds)
            except subprocess.TimeoutExpired:
                stop_process(process)
                result = 124
            raise SystemExit(result)
        finally:
            stop_process(process)
            for fd in pipe_fds:
                os.close(fd)
            fcntl.flock(lock, fcntl.LOCK_UN)
            if lock_started is not None:
                print(f"NPU0 lock released after {time.monotonic()-lock_started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
