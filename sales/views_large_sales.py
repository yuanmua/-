import json
from decimal import Decimal

import pandas as pd
import numpy as np
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

from sales.models_large_sales import GlassProcessingSalesLarge


# 假设直接从数据库获取数据
def get_large_sales_data():
    """
    从数据库获取大型销售数据
    实际项目中应该使用ORM查询
    """
    # 这里应该使用Django ORM查询
    # 例如: GlassProcessingSalesLarge.objects.all()
    # 但由于我们直接使用SQL文件，这里模拟从数据库获取数据
    try:
        # 尝试从CSV文件读取数据（实际项目中应从数据库读取）
        # df = pd.read_csv('sql/glass_processing_sales_data_large.csv')

        queryset = GlassProcessingSalesLarge.objects.all()

        # 转换为DataFrame
        df = pd.DataFrame.from_records(queryset.values(
            'order_id',
            'customer_id',
            'product_name',
            'quantity',
            'unit_price',
            'sale_date',
            'region',
            'total_amount',
        ))

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # 如果无法读取，返回空DataFrame
        return pd.DataFrame()

@login_required
def large_sales_analysis(request):
    """
    大型销售数据分析视图
    """
    # 获取销售数据
    df = get_large_sales_data()

    # 如果数据为空，返回错误信息
    if df.empty:
        return render(request, 'sales/large_sales_analysis.html', {
            'error': '无法加载销售数据，请确保数据已正确导入。'
        })
    
    # 数据预处理
    # 确保日期格式正确
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['month'] = df['sale_date'].dt.month
        df['year'] = df['sale_date'].dt.year
    print(df.head())

    # 1. 按地区统计销售额
    region_sales = df.groupby('region')['total_amount'].sum().reset_index()
    region_sales = region_sales.sort_values('total_amount', ascending=False)
    
    # 2. 按产品统计销售量和销售额
    product_analysis = df.groupby('product_name').agg({
        'quantity': 'sum',
        'total_amount': 'sum'
    }).reset_index()
    product_analysis['avg_price'] = product_analysis['total_amount'] / product_analysis['quantity']
    product_analysis = product_analysis.sort_values('total_amount', ascending=False)
    
    # 3. 按月份统计销售趋势
    if 'month' in df.columns and 'year' in df.columns:
        time_series = df.groupby(['year', 'month'])['total_amount'].sum().reset_index()
        # 新增代码：按时间排序并排除最新月份
        time_series = time_series.sort_values(['year', 'month'])
        if not time_series.empty:
            time_series = time_series.iloc[:-1]  # 移除最后一个月


        time_series['date_label'] = time_series.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}", axis=1)
        
        # 销售预测 - 使用简单线性回归
        if len(time_series) >= 3:  # 至少需要3个数据点
            # 创建时间特征（将年月转换为连续数值）
            time_series['time_index'] = time_series.index
            
            # 训练线性回归模型
            X = time_series[['time_index']]
            y = time_series['total_amount']
            model = LinearRegression()
            model.fit(X, y)
            
            # 预测未来3个月
            future_indices = np.array(range(len(time_series), len(time_series) + 3)).reshape(-1, 1)
            future_sales = model.predict(future_indices)
            
            # 生成未来月份标签
            last_date = pd.to_datetime(f"{int(time_series.iloc[-1]['year'])}-{int(time_series.iloc[-1]['month'])}-01")
            future_dates = []
            for i in range(1, 4):
                next_date = last_date + pd.DateOffset(months=i)
                future_dates.append(f"{next_date.year}-{next_date.month}")
            
            # 预测结果
            predictions = {
                'dates': future_dates,
                'values': future_sales.tolist()
            }
        else:
            predictions = None
    else:
        time_series = pd.DataFrame()
        predictions = None
    
    # 4. 客户分析
    customer_analysis = df.groupby('customer_id').agg({
        'total_amount': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    customer_analysis.columns = ['customer_id', 'total_spending', 'order_count']
    customer_analysis['avg_order_value'] = customer_analysis['total_spending'] / customer_analysis['order_count']
    customer_analysis = customer_analysis.sort_values('total_spending', ascending=False).head(10)
    
    # 准备图表数据
    chart_data = {
        'region_labels': region_sales['region'].tolist(),
        'region_values': region_sales['total_amount'].tolist(),
        'product_labels': product_analysis['product_name'].tolist(),
        'product_quantities': product_analysis['quantity'].tolist(),
        'product_amounts': product_analysis['total_amount'].tolist(),
        'time_labels': time_series['date_label'].tolist() if not time_series.empty else [],
        'time_values': time_series['total_amount'].tolist() if not time_series.empty else [],
        'predictions': predictions,
        'customer_ids': customer_analysis['customer_id'].tolist(),
        'customer_spending': customer_analysis['total_spending'].tolist(),
        'customer_orders': customer_analysis['order_count'].tolist(),
    }
    
    # 计算关键指标
    total_sales = df['total_amount'].sum()
    total_orders = df['order_id'].nunique()
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    total_customers = df['customer_id'].nunique()
    
    # 汇总统计数据
    summary_data = {
        'total_sales': total_sales,
        'total_orders': total_orders,
        'avg_order_value': avg_order_value,
        'total_customers': total_customers,
    }

    # 在返回render之前添加类型转换
    def convert_decimals(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError

    chart_data = json.loads(json.dumps(chart_data, default=convert_decimals))
    summary_data = json.loads(json.dumps(summary_data, default=convert_decimals))
    summary_data = json.loads(json.dumps(summary_data, default=convert_decimals))


    return render(request, 'sales/large_sales_analysis.html', {
        'chart_data': chart_data,
        'summary_data': summary_data,
        'region_sales': region_sales.to_dict('records'),
        'product_analysis': product_analysis.to_dict('records'),
        'customer_analysis': customer_analysis.to_dict('records'),
    })