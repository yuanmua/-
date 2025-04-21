import pandas as pd
import numpy as np
import os
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.db import transaction
from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import Sum
from django.conf import settings

from .forms import UserLoginForm, UserRegisterForm, ProfileForm, UserForm
from sales.models.models_company_sales import GlassProcessingSalesSingleCompany
from sales.models.models_large_sales import GlassProcessingSalesLarge
from sales.ml_models.model_utils import preprocess_data
from sales.ml_models.dashboard_models import get_dashboard_predictor, prepare_product_data


# 这些函数已移至 sales.ml_models.model_utils 模块

def dashboard(request):
    """销售数据仪表盘视图"""
    # 从数据库获取销售数据
    large_sales = GlassProcessingSalesLarge.objects.all()
    company_sales = GlassProcessingSalesSingleCompany.objects.all()
    
    # 将大型销售数据转换为DataFrame
    large_data = {
        'product_name': [item.product_name for item in large_sales],
        'quantity': [item.quantity for item in large_sales],
        'unit_price': [float(item.unit_price) for item in large_sales],
        'total_amount': [float(item.total_amount) for item in large_sales],
        'sale_date': [item.sale_date for item in large_sales],
        'region': [item.region for item in large_sales],
    }
    
    # 将单公司销售数据转换为DataFrame
    company_data = {
        'company_name': [item.company_name for item in company_sales],
        'region': [item.region for item in company_sales],
        'product_name': [item.product_name for item in company_sales],
        'quantity': [item.quantity for item in company_sales],
        'original_price': [float(item.original_price) for item in company_sales],
        'sale_price': [float(item.sale_price) for item in company_sales],
        'sales_amount': [float(item.sales_amount) for item in company_sales],
        'net_margin': [float(item.net_margin) for item in company_sales],
        'sale_date': [item.sale_date for item in company_sales],
    }
    
    # 创建DataFrame
    df_large = pd.DataFrame(large_data)
    df_company = pd.DataFrame(company_data)
    
    # 数据预处理
    if not df_large.empty:
        df_large = preprocess_data(df_large)
    if not df_company.empty:
        df_company = preprocess_data(df_company)
    
    # 合并数据集以获取产品列表
    all_products = set(df_large['product_name'].unique()) | set(df_company['product_name'].unique())
    
    # 计算关键指标
    total_sales = 0
    if not df_large.empty:
        total_sales += df_large['total_amount'].sum()
    if not df_company.empty:
        total_sales += df_company['sales_amount'].sum()
    
    # 计算平均毛利率（从单公司数据）
    avg_gross_margin = 0
    if not df_company.empty and not df_company['sales_amount'].sum() == 0:
        avg_gross_margin = (df_company['net_margin'] * df_company['sales_amount']).sum() / df_company['sales_amount'].sum() * 100
    
    # 按产品类型分组计算销量
    tempered_glass_volume = 0
    laminated_glass_volume = 0
    
    # 从大型销售数据计算
    if not df_large.empty:
        for product in df_large['product_name'].unique():
            if '钢化' in product:
                tempered_glass_volume += df_large[df_large['product_name'] == product]['quantity'].sum()
            elif '夹胶' in product:
                laminated_glass_volume += df_large[df_large['product_name'] == product]['quantity'].sum()
    
    # 从单公司销售数据计算
    if not df_company.empty:
        for product in df_company['product_name'].unique():
            if '钢化' in product:
                tempered_glass_volume += df_company[df_company['product_name'] == product]['quantity'].sum()
            elif '夹胶' in product:
                laminated_glass_volume += df_company[df_company['product_name'] == product]['quantity'].sum()
    
    # 按月份和产品类型分组计算销售额
    month_order = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    monthly_sales_tempered = [0] * 12
    monthly_sales_laminated = [0] * 12
    
    # 从大型销售数据计算月度销售额
    if not df_large.empty:
        for month_idx, month in enumerate(month_order):
            month_num = month_idx + 1
            for product in df_large['product_name'].unique():
                month_product_sales = df_large[(df_large['month'] == month_num) & (df_large['product_name'] == product)]['total_amount'].sum()
                if '钢化' in product:
                    monthly_sales_tempered[month_idx] += float(month_product_sales)
                elif '夹胶' in product:
                    monthly_sales_laminated[month_idx] += float(month_product_sales)
    
    # 从单公司销售数据计算月度销售额
    if not df_company.empty:
        for month_idx, month in enumerate(month_order):
            month_num = month_idx + 1
            for product in df_company['product_name'].unique():
                month_product_sales = df_company[(df_company['month'] == month_num) & (df_company['product_name'] == product)]['sales_amount'].sum()
                if '钢化' in product:
                    monthly_sales_tempered[month_idx] += float(month_product_sales)
                elif '夹胶' in product:
                    monthly_sales_laminated[month_idx] += float(month_product_sales)
    
    # 准备图表数据
    chart_data = {
        'months': month_order,
        '钢化玻璃': monthly_sales_tempered,
        '夹层玻璃': monthly_sales_laminated,
        'values': [total_sales, avg_gross_margin, tempered_glass_volume, laminated_glass_volume]
    }
    
    # 销售预测 - 使用预训练模型
    predictions = {}
    
    try:
        # 获取仪表盘预测器实例 - 直接加载预训练模型
        dashboard_predictor = get_dashboard_predictor()
        
        # 检查模型文件是否存在，如果不存在则预训练
        MODEL_DIR = os.path.join(settings.BASE_DIR, 'sales', 'ml_models', 'saved_models')
        TEMPERED_MODEL_PATH = os.path.join(MODEL_DIR, 'tempered_glass_model.pkl')
        LAMINATED_MODEL_PATH = os.path.join(MODEL_DIR, 'laminated_glass_model.pkl')
        
        if not (os.path.exists(TEMPERED_MODEL_PATH) and os.path.exists(LAMINATED_MODEL_PATH)):
            # 如果模型文件不存在，导入预训练模块并执行预训练
            from sales.ml_models.pretrain_models import pretrain_dashboard_models
            pretrain_dashboard_models()
            # 重新获取预测器实例以加载新训练的模型
            dashboard_predictor = get_dashboard_predictor()
        
        # 对每种产品进行预测
        for product_type in ['钢化玻璃', '夹层玻璃']:
            # 准备产品数据
            product_data = prepare_product_data(df_large, df_company, product_type)
            
            # 确保有足够的数据进行预测
            if not product_data.empty and len(product_data) >= 3:  # 至少需要3个数据点
                # 使用预训练模型预测下一个月
                prediction_result = dashboard_predictor.predict_next_month(product_data, product_type)
                
                if prediction_result:
                    predictions[product_type] = prediction_result
    except Exception as e:
        print(f"预测销售数据时出错: {e}")
        # 出错时不影响页面其他部分的显示
    
    # 将预测数据添加到图表数据中
    chart_data['predictions'] = predictions
    
    # 添加模型评估指标
    chart_data['model_metrics'] = {}
    for product, data in predictions.items():
        if 'model_metrics' in data:
            chart_data['model_metrics'][product] = data['model_metrics']
    
    # 添加材料比例数据（基于销售量计算）
    # 使用销售量作为材料比例的替代指标
    material_ratio_tempered = []
    material_ratio_laminated = []
    
    for month in range(1, 13):
        try:
            # 计算每月钢化玻璃和夹层玻璃的销售量
            tempered_quantity = 0
            laminated_quantity = 0
            
            # 从大型销售数据计算
            if not df_large.empty:
                tempered_month_data = df_large[(df_large['month'] == month) & (df_large['product_name'].str.contains('钢化'))]
                laminated_month_data = df_large[(df_large['month'] == month) & (df_large['product_name'].str.contains('夹胶'))]
                
                if not tempered_month_data.empty:
                    tempered_quantity += tempered_month_data['quantity'].sum()
                if not laminated_month_data.empty:
                    laminated_quantity += laminated_month_data['quantity'].sum()
            
            # 从单公司销售数据计算
            if not df_company.empty:
                tempered_company_data = df_company[(df_company['month'] == month) & (df_company['product_name'].str.contains('钢化'))]
                laminated_company_data = df_company[(df_company['month'] == month) & (df_company['product_name'].str.contains('夹胶'))]
                
                if not tempered_company_data.empty:
                    tempered_quantity += tempered_company_data['quantity'].sum()
                if not laminated_company_data.empty:
                    laminated_quantity += laminated_company_data['quantity'].sum()
            
            # 计算比例并保留两位小数
            total_quantity = tempered_quantity + laminated_quantity
            material_ratio_tempered.append(round(tempered_quantity / total_quantity if total_quantity > 0 else 0, 2))
            material_ratio_laminated.append(round(laminated_quantity / total_quantity if total_quantity > 0 else 0, 2))
        except Exception as e:
            # 发生异常时使用默认值
            print(f"计算材料比例时出错: {e}")
            material_ratio_tempered.append(0.0)
            material_ratio_laminated.append(0.0)
    
    chart_data['material_ratio'] = {
        '钢化玻璃': material_ratio_tempered,
        '夹层玻璃': material_ratio_laminated
    }
    
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