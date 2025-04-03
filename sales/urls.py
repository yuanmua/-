from django.urls import path
from . import views

app_name = 'sales'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),  # 默认显示仪表盘
    path('upload/', views.data_upload, name='data_upload'),  # 数据上传页面
    
    # 用户认证相关URL
    path('login/', views.login_view, name='login'),  # 用户登录
    path('logout/', views.logout_view, name='logout'),  # 用户退出
    path('register/', views.register_view, name='register'),  # 用户注册
    path('profile/', views.profile_view, name='profile'),  # 用户个人资料
    
    # 用户管理相关URL
    path('users/', views.user_management, name='user_management'),  # 用户管理
    path('users/<int:user_id>/toggle-staff/', views.user_toggle_staff, name='user_toggle_staff'),  # 切换管理员状态
    path('users/<int:user_id>/delete/', views.user_delete, name='user_delete'),  # 删除用户
]