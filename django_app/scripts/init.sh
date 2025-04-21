#!/bin/bash
# scripts/init.sh

# 执行数据导入
echo "开始导入数据..."
python3 /app/scripts/import_data.py

# 可选：执行其他初始化操作
# python3 manage.py loaddata initial_data.json
