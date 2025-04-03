#!/usr/bin/env python
import os
import sys

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "无法导入Django。请确保已安装Django。"
            "运行 'pip install django' 安装Django。"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()