#!/usr/bin/env python3
import os
import sys
import time
import socket
import argparse


def check_port(host, port, timeout=2):
    """检查指定端口是否可连接"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError):
        return False


def main():
    parser = argparse.ArgumentParser(description='等待数据库就绪')
    parser.add_argument('host', help='数据库主机')
    parser.add_argument('port', type=int, help='数据库端口')
    parser.add_argument('--timeout', type=int, default=60, help='总超时时间（秒）')
    parser.add_argument('--interval', type=int, default=2, help='检查间隔（秒）')
    args, remaining_args = parser.parse_known_args()

    start_time = time.time()
    print(f"等待数据库 {args.host}:{args.port}...", flush=True)

    while True:
        if check_port(args.host, args.port):
            print(f"数据库 {args.host}:{args.port} 已就绪", flush=True)
            break

        if time.time() - start_time > args.timeout:
            print(f"等待数据库超时 ({args.timeout}秒)", flush=True)
            sys.exit(1)

        print(".", end="", flush=True)
        time.sleep(args.interval)

    # 执行后续命令
    if remaining_args:
        print(f"执行命令: {' '.join(remaining_args)}", flush=True)
        os.execvp(remaining_args[0], remaining_args)


if __name__ == '__main__':
    main()