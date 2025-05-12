from django.urls import path
from . import views, views_large_sales, views_company_sales
from . import views_data_management

app_name = 'sales'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),  # 仪表盘页面
    path('upload/', views_data_management.data_upload, name='data_upload'),  # 数据上传页面
    
    # 用户认证相关URL
    path('login/', views.login_view, name='login'),  # 登录页面
    path('logout/', views.logout_view, name='logout'),  # 登出
    path('register/', views.register_view, name='register'),  # 注册页面
    path('profile/', views.profile_view, name='profile'),  # 个人资料页面
    
    # 用户管理相关URL
    path('users/', views.user_management, name='user_management'),  # 用户管理
    path('users/<int:user_id>/toggle-staff/', views.user_toggle_staff, name='user_toggle_staff'),  # 切换管理员状态
    path('users/<int:user_id>/delete/', views.user_delete, name='user_delete'),  # 删除用户

    # 销售分析相关URL
    path('views_large_sales/', views_large_sales.large_sales_analysis, name='large_sales_analysis'),  # 大型销售分析
    path('views_company_sales/', views_company_sales.company_sales_analysis, name='company_sales_analysis'),  # 单公司销售分析


    # 大型玻璃加工销售数据管理
    path('large-sales/', views_data_management.LargeSalesListView.as_view(), name='large_sales_list'),
    path('large-sales/<int:pk>/', views_data_management.LargeSalesDetailView.as_view(), name='large_sales_detail'),
    path('large-sales/<int:pk>/edit/', views_data_management.LargeSalesUpdateView.as_view(), name='large_sales_edit'),
    path('large-sales/<int:pk>/delete/', views_data_management.LargeSalesDeleteView.as_view(), name='large_sales_delete'),
    path('large-sales/export-csv/', views_data_management.export_large_sales_csv, name='export_large_sales_csv'),

    # 单公司玻璃销售数据管理
    path('single-company/', views_data_management.SingleCompanyListView.as_view(), name='single_company_list'),
    path('single-company/<int:pk>/', views_data_management.SingleCompanyDetailView.as_view(), name='single_company_detail'),
    path('single-company/<int:pk>/edit/', views_data_management.SingleCompanyUpdateView.as_view(), name='single_company_edit'),
    path('single-company/<int:pk>/delete/', views_data_management.SingleCompanyDeleteView.as_view(), name='single_company_delete'),
    path('single-company/export-csv/', views_data_management.export_single_company_csv, name='export_single_company_csv'),
]