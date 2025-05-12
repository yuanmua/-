from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.contrib import messages
from django.db.models import Count, Sum, Avg, F
from django.http import HttpResponse, JsonResponse
import pandas as pd
import json
import csv
import io
from datetime import datetime

from .models.models_large_sales import GlassProcessingSalesLarge
from .models.models_single_company import GlassProcessingSalesSingleCompany

# 数据上传处理视图
def data_upload(request):
    """数据上传视图"""
    if request.method == 'POST':
        data_file = request.FILES.get('data_file')
        data_type = request.POST.get('data_type')
        
        if not data_file:
            messages.error(request, '请选择要上传的文件')
            return render(request, 'sales/data_upload.html')
            
        if not data_type:
            messages.error(request, '请选择数据类型')
            return render(request, 'sales/data_upload.html')
        
        # 根据文件扩展名处理数据
        file_name = data_file.name.lower()
        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(data_file)
            elif file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(data_file)
            else:
                messages.error(request, '不支持的文件格式，请上传CSV或Excel文件')
                return render(request, 'sales/data_upload.html')
                
            # 根据数据类型处理不同的表
            if data_type == 'large_sales':
                process_large_sales_data(df, request)
                return redirect('sales:large_sales_list')
            elif data_type == 'single_company':
                process_single_company_data(df, request)
                return redirect('sales:single_company_list')
            else:
                messages.error(request, '不支持的数据类型')
                
        except Exception as e:
            messages.error(request, f'数据处理错误: {str(e)}')
    
    return render(request, 'sales/data_upload.html')

# 处理大型玻璃加工销售数据
def process_large_sales_data(df, request):
    # 检查必要的列是否存在
    required_columns = ['订单ID', '客户ID', '产品名称', '数量', '单价', '销售日期', '地区', '总金额']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f'缺少必要的列: {", ".join(missing_columns)}')
    
    # 列名映射
    column_mapping = {
        '订单ID': 'order_id',
        '客户ID': 'customer_id',
        '产品名称': 'product_name',
        '数量': 'quantity',
        '单价': 'unit_price',
        '销售日期': 'sale_date',
        '地区': 'region',
        '总金额': 'total_amount'
    }
    
    # 重命名列
    df = df.rename(columns=column_mapping)
    
    # 转换日期格式
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    
    # 创建模型实例并保存
    records_created = 0
    for _, row in df.iterrows():
        GlassProcessingSalesLarge.objects.create(
            order_id=row['order_id'],
            customer_id=row['customer_id'],
            product_name=row['product_name'],
            quantity=row['quantity'],
            unit_price=row['unit_price'],
            sale_date=row['sale_date'],
            region=row['region'],
            total_amount=row['total_amount']
        )
        records_created += 1
    
    messages.success(request, f'成功导入 {records_created} 条大型玻璃加工销售数据')

# 处理单公司玻璃销售数据
def process_single_company_data(df, request):
    # 检查必要的列是否存在
    required_columns = ['公司名称', '地区', '产品名称', '数量', '原始价格', '销售价格', '销售金额', '净利润率', '销售日期']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f'缺少必要的列: {", ".join(missing_columns)}')
    
    # 列名映射
    column_mapping = {
        '公司名称': 'company_name',
        '地区': 'region',
        '产品名称': 'product_name',
        '数量': 'quantity',
        '原始价格': 'original_price',
        '销售价格': 'sale_price',
        '销售金额': 'sales_amount',
        '净利润率': 'net_margin',
        '销售日期': 'sale_date'
    }
    
    # 重命名列
    df = df.rename(columns=column_mapping)
    
    # 转换日期格式
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    
    # 创建模型实例并保存
    records_created = 0
    for _, row in df.iterrows():
        GlassProcessingSalesSingleCompany.objects.create(
            company_name=row['company_name'],
            region=row['region'],
            product_name=row['product_name'],
            quantity=row['quantity'],
            original_price=row['original_price'],
            sale_price=row['sale_price'],
            sales_amount=row['sales_amount'],
            net_margin=row['net_margin'],
            sale_date=row['sale_date']
        )
        records_created += 1
    
    messages.success(request, f'成功导入 {records_created} 条单公司玻璃销售数据')

# 大型玻璃加工销售数据列表视图
class LargeSalesListView(ListView):
    model = GlassProcessingSalesLarge
    template_name = 'sales/large_sales_list.html'
    context_object_name = 'sales_data'
    paginate_by = 10
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # 统计数据
        context['total_orders'] = GlassProcessingSalesLarge.objects.count()
        context['total_amount'] = GlassProcessingSalesLarge.objects.aggregate(Sum('total_amount'))['total_amount__sum'] or 0
        context['total_customers'] = GlassProcessingSalesLarge.objects.values('customer_id').distinct().count()
        context['total_products'] = GlassProcessingSalesLarge.objects.values('product_name').distinct().count()
        
        # 产品销售分布数据
        product_data = GlassProcessingSalesLarge.objects.values('product_name').annotate(
            total=Sum('total_amount')
        ).order_by('-total')[:10]
        
        context['product_labels'] = json.dumps([item['product_name'] for item in product_data])
        context['product_data'] = json.dumps([float(item['total']) for item in product_data])
        
        # 销售趋势数据
        trend_data = GlassProcessingSalesLarge.objects.values('sale_date').annotate(
            total=Sum('total_amount')
        ).order_by('sale_date')[:30]
        
        context['date_labels'] = json.dumps([item['sale_date'].strftime('%Y-%m-%d') for item in trend_data])
        context['trend_data'] = json.dumps([float(item['total']) for item in trend_data])
        
        return context

# 单公司玻璃销售数据列表视图
class SingleCompanyListView(ListView):
    model = GlassProcessingSalesSingleCompany
    template_name = 'sales/single_company_list.html'
    context_object_name = 'sales_data'
    paginate_by = 10
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # 统计数据
        context['total_companies'] = GlassProcessingSalesSingleCompany.objects.values('company_name').distinct().count()
        context['total_amount'] = GlassProcessingSalesSingleCompany.objects.aggregate(Sum('sales_amount'))['sales_amount__sum'] or 0
        context['avg_margin'] = GlassProcessingSalesSingleCompany.objects.aggregate(Avg('net_margin'))['net_margin__avg'] or 0
        context['total_products'] = GlassProcessingSalesSingleCompany.objects.values('product_name').distinct().count()
        
        # 公司销售分布数据
        company_data = GlassProcessingSalesSingleCompany.objects.values('company_name').annotate(
            total=Sum('sales_amount')
        ).order_by('-total')[:10]
        
        context['company_labels'] = json.dumps([item['company_name'] for item in company_data])
        context['company_data'] = json.dumps([float(item['total']) for item in company_data])
        
        # 利润率趋势数据
        margin_data = GlassProcessingSalesSingleCompany.objects.values('sale_date').annotate(
            avg_margin=Avg('net_margin')
        ).order_by('sale_date')[:30]
        
        context['date_labels'] = json.dumps([item['sale_date'].strftime('%Y-%m-%d') for item in margin_data])
        context['margin_data'] = json.dumps([float(item['avg_margin']) for item in margin_data])
        
        return context

# 大型玻璃加工销售数据详情视图
class LargeSalesDetailView(DetailView):
    model = GlassProcessingSalesLarge
    template_name = 'sales/large_sales_detail.html'
    context_object_name = 'sales_item'

# 单公司玻璃销售数据详情视图
class SingleCompanyDetailView(DetailView):
    model = GlassProcessingSalesSingleCompany
    template_name = 'sales/single_company_detail.html'
    context_object_name = 'sales_item'

# 大型玻璃加工销售数据编辑视图
class LargeSalesUpdateView(UpdateView):
    model = GlassProcessingSalesLarge
    template_name = 'sales/large_sales_form.html'
    fields = ['order_id', 'customer_id', 'product_name', 'quantity', 'unit_price', 'sale_date', 'region', 'total_amount']
    success_url = reverse_lazy('sales:large_sales_list')
    
    def form_valid(self, form):
        messages.success(self.request, '销售数据更新成功')
        return super().form_valid(form)

# 单公司玻璃销售数据编辑视图
class SingleCompanyUpdateView(UpdateView):
    model = GlassProcessingSalesSingleCompany
    template_name = 'sales/single_company_form.html'
    fields = ['company_name', 'region', 'product_name', 'quantity', 'original_price', 'sale_price', 'sales_amount', 'net_margin', 'sale_date']
    success_url = reverse_lazy('sales:single_company_list')
    
    def form_valid(self, form):
        messages.success(self.request, '销售数据更新成功')
        return super().form_valid(form)

# 大型玻璃加工销售数据删除视图
class LargeSalesDeleteView(DeleteView):
    model = GlassProcessingSalesLarge
    success_url = reverse_lazy('sales:large_sales_list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, '销售数据删除成功')
        return super().delete(request, *args, **kwargs)

# 单公司玻璃销售数据删除视图
class SingleCompanyDeleteView(DeleteView):
    model = GlassProcessingSalesSingleCompany
    success_url = reverse_lazy('sales:single_company_list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, '销售数据删除成功')
        return super().delete(request, *args, **kwargs)

# 导出大型玻璃加工销售数据为CSV
def export_large_sales_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="large_sales_data.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['订单ID', '客户ID', '产品名称', '数量', '单价', '销售日期', '地区', '总金额'])
    
    sales_data = GlassProcessingSalesLarge.objects.all().values_list(
        'order_id', 'customer_id', 'product_name', 'quantity', 'unit_price', 'sale_date', 'region', 'total_amount'
    )
    for item in sales_data:
        writer.writerow(item)
    
    return response

# 导出单公司玻璃销售数据为CSV
def export_single_company_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="single_company_data.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['公司名称', '地区', '产品名称', '数量', '原始价格', '销售价格', '销售金额', '净利润率', '销售日期'])
    
    sales_data = GlassProcessingSalesSingleCompany.objects.all().values_list(
        'company_name', 'region', 'product_name', 'quantity', 'original_price', 'sale_price', 'sales_amount', 'net_margin', 'sale_date'
    )
    for item in sales_data:
        writer.writerow(item)
    
    return response