import json
from decimal import Decimal

import pandas as pd
import numpy as np
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

from sales.models import SalesData
from sales.models_company_sales import GlassProcessingSalesSingleCompany


# 假设直接从数据库获取数据
def get_company_sales_data():
    """
    从数据库获取单公司销售数据
    实际项目中应该使用ORM查询
    """
    # 这里应该使用Django ORM查询
    # 例如: GlassProcessingSalesSingleCompany.objects.all()
    # 但由于我们直接使用SQL文件，这里模拟从数据库获取数据
    try:
        # 尝试从CSV文件读取数据（实际项目中应从数据库读取）
        # 使用ORM查询所有数据
        queryset = GlassProcessingSalesSingleCompany.objects.all()

        # 转换为DataFrame
        df = pd.DataFrame.from_records(queryset.values(
            'company_name',
            'region',
            'product_name',
            'quantity',
            'original_price',
            'sale_price',
            'sales_amount',
            'net_margin',
            'sale_date'
        ))

        # 类型转换（如果需要）
        # if not df.empty:
        #     df['sale_date'] = pd.to_datetime(df['sale_date'])

        # df = pd.read_csv('sql/glass_processing_sales_data_single_company.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # 如果无法读取，返回空DataFrame
        return pd.DataFrame()

@login_required
def company_sales_analysis(request):
    """
    单公司销售数据分析视图
    """
    # 获取销售数据
    df = get_company_sales_data()

    # 如果数据为空，返回错误信息
    if df.empty:
        return render(request, 'sales/company_sales_analysis.html', {
            'error': '无法加载公司销售数据，请确保数据已正确导入。'
        })
    # 数据预处理
    # 确保日期格式正确
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['month'] = df['sale_date'].dt.month
        df['year'] = df['sale_date'].dt.year

    # 1. 公司销售概览
    company_overview = df.groupby('company_name').agg({
        'sales_amount': 'sum',
        'quantity': 'sum',
        'net_margin': lambda x: (x * df.loc[x.index, 'sales_amount']).sum() / df.loc[x.index, 'sales_amount'].sum()
    }).reset_index()
    company_overview.columns = ['company_name', 'total_sales', 'total_quantity', 'avg_margin']
    company_overview = company_overview.sort_values('total_sales', ascending=False)

    # 2. 按地区分析
    region_analysis = df.groupby(['company_name', 'region']).agg({
        'sales_amount': 'sum'
    }).reset_index()

    # 获取销售额最高的公司
    top_company = company_overview.iloc[0]['company_name'] if not company_overview.empty else None

    # 3. 产品分析
    product_analysis = df.groupby(['company_name', 'product_name']).agg({
        'quantity': 'sum',
        'sales_amount': 'sum',
        'net_margin': 'mean'
    }).reset_index()

    # 4. 时间序列分析和预测
    if 'month' in df.columns and 'year' in df.columns and top_company:
        # 筛选出销售额最高的公司数据
        top_company_data = df[df['company_name'] == top_company]
        time_series = top_company_data.groupby(['year', 'month'])['sales_amount'].sum().reset_index()

        # 新增代码：按时间排序并排除最新月份
        time_series = time_series.sort_values(['year', 'month'])
        if not time_series.empty:
            time_series = time_series.iloc[:-1]  # 移除最后一个月
        time_series['date_label'] = time_series.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}", axis=1)



        time_series['date_label'] = time_series.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}", axis=1)

        # 销售预测 - 使用简单线性回归
        if len(time_series) >= 3:  # 至少需要3个数据点

            # 创建时间特征
            time_series['time_index'] = time_series.index

            # 训练线性回归模型
            X = time_series[['time_index']]
            y = time_series['sales_amount']
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
                'company': top_company,
                'dates': future_dates,
                'values': future_sales.tolist()
            }
        else:
            predictions = None
    else:
        time_series = pd.DataFrame()
        predictions = None

    # 5. 价格分析
    price_analysis = df.groupby(['company_name', 'product_name']).agg({
        'original_price': 'mean',
        'sale_price': 'mean'
    }).reset_index()
    price_analysis['discount_rate'] = (price_analysis['original_price'] - price_analysis['sale_price']) / price_analysis['original_price'] * 100

    # 准备图表数据
    chart_data = {
        'company_names': company_overview['company_name'].tolist(),
        'company_sales': company_overview['total_sales'].tolist(),
        'company_margins': company_overview['avg_margin'].tolist(),
        'time_labels': time_series['date_label'].tolist() if not time_series.empty else [],
        'time_values': time_series['sales_amount'].tolist() if not time_series.empty else [],
        'predictions': predictions,
    }

    # 计算关键指标
    total_sales = df['sales_amount'].sum()
    avg_margin = (df['net_margin'] * df['sales_amount']).sum() / total_sales if total_sales > 0 else 0
    total_companies = df['company_name'].nunique()

    # 汇总统计数据
    summary_data = {
        'total_sales': total_sales,
        'avg_margin': avg_margin,
        'total_companies': total_companies,
        'top_company': top_company,
    }

    # 在返回render之前添加类型转换
    def convert_decimals(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError

    chart_data = json.loads(json.dumps(chart_data, default=convert_decimals))
    summary_data = json.loads(json.dumps(summary_data, default=convert_decimals))


    return render(request, 'sales/company_sales_analysis.html', {
        'chart_data': chart_data,
        'summary_data': summary_data,
        'company_overview': company_overview.to_dict('records'),
        'product_analysis': product_analysis.to_dict('records'),
        'price_analysis': price_analysis.to_dict('records'),
    })