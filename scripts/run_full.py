"""Print a pipeline by default; execution requires --run on the destination machine."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/full.json')
    parser.add_argument('--output', default='runs/full')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    locked = (output / 'LOCK.json').exists()
    stages = ['download'] + ([] if locked else ['tune', 'lock']) + ['fit', 'evaluate', 'analyze']
    for stage in stages:
        command = [sys.executable, '-m', 'mla.cli', stage, '--config', args.config,
                   '--output', str(output), '--device', args.device, '--threads', str(args.threads)]
        print(subprocess.list2cmdline(command), flush=True)
        if args.run:
            subprocess.run(command, cwd=root, check=True)
    if not args.run:
        print('Preview only. Add --run to download data and execute the experiments.')


if __name__ == '__main__':
    main()
