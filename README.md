# 项目结构(可能)
```
glass_analysis/
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── sales/
│   ├── migrations/
│   ├── templates/
│   │   └── sales/
│   │       ├── base.html
│   │       ├── dashboard.html
│   │       └── data_upload.html
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── utils.py
│   ├── views.py
│   └── urls.py
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── charts.js
├── requirements.txt
└── manage.py
```

如果你不用docker的话启动麻烦得很
我换一台电脑都要折腾一个小时
配数据库
安装包 mysqlclient等等
导入数据
初始化Djang用户配置数据库 Python manage.py migrate
运行 python manage.py migrate


用docker
```bash
docker-compose down

```
```bash
docker-compose build

```
```bash
docker-compose up -d
```

docker-compose up -d --build
