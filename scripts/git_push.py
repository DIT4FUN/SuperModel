#!/usr/bin/env python3
"""Git push script for SuperModel"""
import subprocess
import sys

commands = [
    ["git", "init"],
    ["git", "config", "user.email", "supermodel@dit4fun.github"],
    ["git", "config", "user.name", "SuperModel Bot"],
    ["git", "add", "-A"],
    ["git", "commit", "-m", "feat: 添加触觉/力觉/IMU传感器模块、控制模块、测试用例"],
    ["git", "remote", "add", "origin", "https://github.com/DIT4FUN/SuperModel.git"],
    ["git", "push", "-u", "origin", "main", "--force"],
]

repo = "/home/treeman/.openclaw/workspace/projects/SuperModel"

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=30)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0 and "git init" not in cmd and "remote" not in cmd[0]:
        print(f"Command failed with code {result.returncode}")
        sys.exit(1)

print("Git push completed!")
