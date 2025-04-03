from django.urls import path
from . import views

app_name = 'sales'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),  # 默认显示仪表盘
    path('upload/', views.data_upload, name='data_upload'),  # 数据上传页面
]