import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    print(
        "Missing dependency: psutil\n"
        "Install it with: python -m pip install psutil",
        file=sys.stderr,
    )
    raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a command and record CPU/RAM usage for the process tree."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between measurements.",
    )
    parser.add_argument(
        "--csv",
        default="usage_log.csv",
        help="Path for per-sample CPU/RAM measurements.",
    )
    parser.add_argument(
        "--summary",
        default="usage_summary.json",
        help="Path for final summary metrics.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Put it after --, for example: -- python detect.py ...",
    )
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    if not args.command:
        parser.error("missing command to run after --")

    return args


def process_tree(root):
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass
    return processes


def sample_usage(root):
    total_rss = 0
    total_cpu = 0.0
    process_count = 0

    for proc in process_tree(root):
        try:
            with proc.oneshot():
                total_rss += proc.memory_info().rss
                total_cpu += proc.cpu_percent(interval=None)
                process_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return {
        "cpu_percent": total_cpu,
        "ram_mb": total_rss / (1024 * 1024),
        "process_count": process_count,
    }


def terminate_tree(root):
    children = root.children(recursive=True)
    for proc in children:
        try:
            proc.terminate()
        except psutil.Error:
            pass

    try:
        root.terminate()
    except psutil.Error:
        pass

    gone, alive = psutil.wait_procs(children + [root], timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    summary_path = Path(args.summary)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    popen = subprocess.Popen(args.command)
    root = psutil.Process(popen.pid)

    # Prime psutil CPU counters for every process that exists at startup.
    for proc in process_tree(root):
        try:
            proc.cpu_percent(interval=None)
        except psutil.Error:
            pass

    samples = []
    return_code = None

    try:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "elapsed_seconds",
                    "cpu_percent",
                    "ram_mb",
                    "process_count",
                ],
            )
            writer.writeheader()

            while True:
                return_code = popen.poll()
                usage = sample_usage(root)
                elapsed = time.time() - started_at
                row = {
                    "elapsed_seconds": round(elapsed, 3),
                    "cpu_percent": round(usage["cpu_percent"], 2),
                    "ram_mb": round(usage["ram_mb"], 2),
                    "process_count": usage["process_count"],
                }
                writer.writerow(row)
                handle.flush()
                samples.append(row)

                if return_code is not None:
                    break

                time.sleep(args.interval)
    except KeyboardInterrupt:
        terminate_tree(root)
        return_code = 130
    finally:
        elapsed = time.time() - started_at

    if samples:
        max_ram_mb = max(row["ram_mb"] for row in samples)
        max_cpu_percent = max(row["cpu_percent"] for row in samples)
        avg_cpu_percent = sum(row["cpu_percent"] for row in samples) / len(samples)
    else:
        max_ram_mb = 0.0
        max_cpu_percent = 0.0
        avg_cpu_percent = 0.0

    summary = {
        "command": args.command,
        "return_code": return_code,
        "elapsed_seconds": round(elapsed, 3),
        "sample_interval_seconds": args.interval,
        "samples": len(samples),
        "max_ram_mb": round(max_ram_mb, 2),
        "max_cpu_percent": round(max_cpu_percent, 2),
        "avg_cpu_percent": round(avg_cpu_percent, 2),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nUsage summary")
    print(f"  elapsed: {summary['elapsed_seconds']} seconds")
    print(f"  peak RAM: {summary['max_ram_mb']} MB")
    print(f"  max CPU: {summary['max_cpu_percent']}%")
    print(f"  avg CPU: {summary['avg_cpu_percent']}%")
    print(f"  CSV log: {csv_path}")
    print(f"  summary: {summary_path}")

    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
