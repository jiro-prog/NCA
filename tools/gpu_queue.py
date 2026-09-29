"""
GPU の学習を1本ずつ順に回す列（CLAUDE.md の作業の約束：学習は1本ずつ）。

    .venv/Scripts/python.exe tools/gpu_queue.py          # 列の番をする（バックグラウンドで回しておく）

- 列は queue/queue.txt。1行に1本：  名前<TAB>ログの置き場所<TAB>コマンド
  - 上から順に、まだ終わっていない最初の行を回す。行は回している途中でも足せる（割り込ませたいときは上のほうに書く）
  - # で始まる行は飛ばす。STOP だけの行に来たら（それより上が全部終わっていれば）番をやめる
- 終わった記録は queue/done.txt（名前、終了コード、始まり、終わり、秒）。同じ名前は2回回さない
- 列が空になったら、60秒ごとに見直す
"""
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "queue"
QUEUE = QDIR / "queue.txt"
DONE = QDIR / "done.txt"


def done_names():
    if not DONE.exists():
        return set()
    return {line.split("\t")[0] for line in DONE.read_text(encoding="utf-8").splitlines() if line.strip()}


def next_job():
    """(名前, ログ, コマンド) か、"STOP"、None（今は回すものがない）"""
    if not QUEUE.exists():
        return None
    done = done_names()
    for line in QUEUE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s == "STOP":
            return "STOP"
        parts = line.split("\t")
        if len(parts) != 3:
            print(f"読めない行を飛ばす：{line!r}", flush=True)
            continue
        name, log, cmd = (p.strip() for p in parts)
        if name not in done:
            return name, log, cmd
    return None


def main():
    QDIR.mkdir(exist_ok=True)
    print(f"列の番を始める：{QUEUE}", flush=True)
    while True:
        job = next_job()
        if job == "STOP":
            print("STOP に来たので、番をやめる", flush=True)
            return
        if job is None:
            time.sleep(60)
            continue
        name, log, cmd = job
        log_path = ROOT / log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = datetime.now()
        print(f"[{start:%m-%d %H:%M}] 始める：{name}", flush=True)
        argv = shlex.split(cmd, posix=True)
        exe = ROOT / argv[0]
        if exe.exists():            # Windows は相対パスの実行ファイルを cwd から探さないので、絶対パスにする
            argv[0] = str(exe)
        with open(log_path, "w", encoding="utf-8") as f:
            try:
                code = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
            except OSError as err:  # 起動できなかった行は、終了コード −1 として記録して次へ
                f.write(f"起動できなかった：{err}\n")
                code = -1
        end = datetime.now()
        secs = (end - start).total_seconds()
        with open(DONE, "a", encoding="utf-8") as f:
            f.write(f"{name}\t{code}\t{start:%Y-%m-%d %H:%M:%S}\t{end:%Y-%m-%d %H:%M:%S}\t{secs:.0f}\n")
        print(f"[{end:%m-%d %H:%M}] 終わった：{name}（終了コード {code}、{secs / 60:.1f}分）", flush=True)


if __name__ == "__main__":
    sys.exit(main())
