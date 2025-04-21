#FROM python:3.10-slim
FROM python:3

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 安装基础依赖
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends netcat && \
#    rm -rf /var/lib/apt/lists/*

# 先安装依赖保证缓存有效
COPY requirements.txt .
RUN apt-get update && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install pymysql pandas sqlalchemy cryptography

# 复制所有必要文件
COPY . .
COPY ./data/*.csv /app/data/
COPY ./scripts/* /app/scripts/

# 设置执行权限
RUN chmod +x /app/scripts/init.sh

# 自定义入口点
COPY ./scripts/wait-for-db.sh .
RUN chmod +x wait-for-db.sh

#CMD ["./wait-for-db.sh", "db:3306", "--", "sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]
CMD ["/app/scripts/init.sh"]
CMD ["sh", "-c", "python ./project/manage.py migrate && python ./project/manage.py runserver 0.0.0.0:8000"]
