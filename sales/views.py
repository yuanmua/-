from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from .models import SalesData

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
    
    # 准备图表数据
    chart_data = {
        'months': month_order,
        '钢化玻璃': monthly_sales.get('钢化玻璃', [0] * 12),
        '夹胶玻璃': monthly_sales.get('夹胶玻璃', [0] * 12),
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