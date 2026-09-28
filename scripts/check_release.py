#!/usr/bin/env python3
"""Fail closed on common accidental experiment artifacts in reachable history.

This is a release hygiene check, not a comprehensive secret scanner. Private
copyright-author emails in notices are not automatically classified as secrets.
"""
from pathlib import PurePosixPath
import re
import subprocess


def git(*args):
    return subprocess.check_output(['git', *args])


def main():
    commits = git('rev-list', '--all').decode().splitlines()
    if not commits:
        raise SystemExit('No committed release to check')
    seen = set()
    violations = []
    for commit in commits:
        for entry in git('ls-tree', '-rz', commit).split(b'\0'):
            if not entry:
                continue
            attributes, path_bytes = entry.split(b'\t', 1)
            _, kind, oid = attributes.split()
            path = path_bytes.decode('utf8')
            parts = PurePosixPath(path).parts
            if any(p in {'wandb', 'outputs', 'eval_results', '__pycache__', '.venv'} for p in parts) or path.endswith(('.pt', '.pth', '.ckpt', '.log', '.wandb', '.bundle', '.zip')):
                violations.append((path, 'experiment/archive artifact'))
            if kind != b'blob' or oid in seen:
                continue
            seen.add(oid)
            content = git('cat-file', 'blob', oid.decode())
            patterns = {
                'machine identifier': rb'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{20,}',
                'private home directory': rb'/(?:home|Users)/[a-zA-Z0-9_.-]+/',
                'private key': rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
                'access token': rb'\b(?:ghp_[a-zA-Z0-9]{30,}|github_pat_[a-zA-Z0-9_]{40,})\b',
            }
            for name, pattern in patterns.items():
                if re.search(pattern, content):
                    violations.append((path, name))
    if violations:
        for path, category in sorted(set(violations)):
            print(f'{path}: {category}')  # Never print matching private values.
        raise SystemExit('Release hygiene check failed')
    print(f'Release hygiene passed: {len(commits)} commits, {len(seen)} unique blobs; no raw experiment artifacts.')


if __name__ == '__main__':
    main()
