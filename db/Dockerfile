# 使用MySQL 8.0官方镜像
FROM mysql:8.0

# 设置环境变量
ENV MYSQL_ROOT_PASSWORD=123456
ENV MYSQL_DATABASE=glass_analysis
ENV MYSQL_USER=django_user
ENV MYSQL_PASSWORD=django_password


# 复制数据库初始化文件
COPY init.sql /docker-entrypoint-initdb.d/01_schema.sql

