#!/bin/sh

# 等待数据库就绪
python scripts/wait_for_db.py

# 执行数据库迁移
python project/manage.py migrate

# 执行数据导入
echo "开始导入数据..."
python3 /app/scripts/import_data.py

python scripts/sjk.py

# 启动应用
exec python project/manage.py runserver 0.0.0.0:8000