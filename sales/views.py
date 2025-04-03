from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.db import transaction
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from .models import SalesData, Profile
from .forms import UserLoginForm, UserRegisterForm, ProfileForm, UserForm

def dashboard(request):
    """销售数据仪表盘视图"""
    # 从数据库获取所有销售数据
    sales_data = SalesData.objects.all()
    
    # 将数据转换为DataFrame进行分析
    data = {
        'month': [item.month for item in sales_data],
        'product': [item.product for item in sales_data],
        'sales_volume': [item.sales_volume for item in sales_data],
        'sales_amount': [item.sales_amount for item in sales_data],
        'cost': [item.cost for item in sales_data],
        'gross_margin': [item.gross_margin for item in sales_data],
        'unit_price': [item.unit_price for item in sales_data],
        'material_ratio': [item.material_ratio for item in sales_data],
        'major_client': [item.major_client for item in sales_data],
    }
    
    df = pd.DataFrame(data)
    
    # 计算关键指标
    total_sales = df['sales_amount'].sum()
    avg_gross_margin = (df['gross_margin'] * df['sales_amount']).sum() / total_sales * 100
    
    # 按产品类型分组计算销量
    product_volume = df.groupby('product')['sales_volume'].sum()
    tempered_glass_volume = product_volume.get('钢化玻璃', 0)
    laminated_glass_volume = product_volume.get('夹胶玻璃', 0)
    
    # 按月份和产品类型分组计算销售额
    monthly_sales = df.pivot_table(index='month', columns='product', values='sales_amount', aggfunc='sum').fillna(0)
    
    # 确保所有月份都有数据，按月份顺序排序
    month_order = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    monthly_sales = monthly_sales.reindex(month_order)
    monthly_sales = monthly_sales.fillna(0)  # 将NaN替换为0

    # 准备图表数据
    chart_data = {
        'months': month_order,
        '钢化玻璃': monthly_sales.get('钢化玻璃', pd.Series([0]*12)).tolist(),
        '夹胶玻璃': monthly_sales.get('夹胶玻璃', pd.Series([0]*12)).tolist(),
        'values': [total_sales, avg_gross_margin, tempered_glass_volume, laminated_glass_volume]
    }
    
    # 销售预测
    # 将月份转换为数值型特征（1-12）
    month_to_num = {month: i+1 for i, month in enumerate(month_order)}
    df['month_num'] = df['month'].map(month_to_num)
    
    # 分产品类型进行预测
    predictions = {}
    for product in df['product'].unique():
        product_data = df[df['product'] == product]
        if len(product_data) >= 3:  # 至少需要3个数据点才能进行预测
            X = product_data[['month_num']]
            y = product_data['sales_amount']
            model = LinearRegression()
            model.fit(X, y)
            
            # 预测下一个月的销售额
            next_month = (product_data['month_num'].max() % 12) + 1
            predicted_sales = model.predict([[next_month]])[0]
            predictions[product] = {
                'next_month': month_order[next_month-1],
                'predicted_sales': round(predicted_sales, 2)
            }
    
    # 将预测数据添加到图表数据中
    chart_data['predictions'] = predictions
    
    return render(request, 'sales/dashboard.html', {'chart_data': chart_data})

def data_upload(request):
    """数据上传视图"""
    return render(request, 'sales/data_upload.html')


def login_view(request):
    """用户登录视图"""
    if request.user.is_authenticated:
        return redirect('sales:dashboard')
        
    if request.method == 'POST':
        form = UserLoginForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'欢迎回来，{username}！')
                return redirect('sales:dashboard')
        else:
            messages.error(request, '登录失败，请检查用户名和密码。')
    else:
        form = UserLoginForm()
    return render(request, 'sales/login.html', {'form': form})


def register_view(request):
    """用户注册视图"""
    if request.user.is_authenticated:
        return redirect('sales:dashboard')
        
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'账号创建成功，{username}！请登录。')
            return redirect('sales:login')
    else:
        form = UserRegisterForm()
    return render(request, 'sales/register.html', {'form': form})


def logout_view(request):
    """用户退出登录视图"""
    logout(request)
    messages.info(request, '您已成功退出登录。')
    return redirect('sales:login')


@login_required
def profile_view(request):
    """用户个人资料视图"""
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, instance=request.user.profile)
        if user_form.is_valid() and profile_form.is_valid():
            with transaction.atomic():
                user_form.save()
                profile_form.save()
            messages.success(request, '个人资料已更新！')
            return redirect('sales:profile')
    else:
        user_form = UserForm(instance=request.user)
        profile_form = ProfileForm(instance=request.user.profile)
    
    return render(request, 'sales/profile.html', {
        'user_form': user_form,
        'profile_form': profile_form
    })


@login_required
@user_passes_test(lambda u: u.is_staff)
def user_management(request):
    """用户管理视图，仅管理员可访问"""
    users = User.objects.all().order_by('-date_joined')
    return render(request, 'sales/user_management.html', {'users': users})


@login_required
@user_passes_test(lambda u: u.is_staff)
def user_toggle_staff(request, user_id):
    """切换用户的管理员状态"""
    if request.user.id == user_id:
        messages.error(request, '不能修改自己的管理员状态！')
        return redirect('sales:user_management')
        
    user = get_object_or_404(User, id=user_id)
    user.is_staff = not user.is_staff
    user.save()
    messages.success(request, f'已{"授予" if user.is_staff else "撤销"}{user.username}的管理员权限。')
    return redirect('sales:user_management')


@login_required
@user_passes_test(lambda u: u.is_staff)
def user_delete(request, user_id):
    """删除用户"""
    if request.user.id == user_id:
        messages.error(request, '不能删除自己的账号！')
        return redirect('sales:user_management')
        
    user = get_object_or_404(User, id=user_id)
    username = user.username
    user.delete()
    messages.success(request, f'用户 {username} 已被删除。')
    return redirect('sales:user_management')