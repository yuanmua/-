import os
import pandas as pd
from django.db.models import Sum
from django.conf import settings

from sales.models.models_single_company import GlassProcessingSalesSingleCompany
from sales.models.models_large_sales import GlassProcessingSalesLarge
from sales.ml_models.model_utils import preprocess_data
from sales.ml_models.large_sales_models import get_large_sales_predictor, prepare_time_series_data
from sales.ml_models.company_sales_models import get_company_sales_predictor, prepare_company_time_series_data
from sales.ml_models.dashboard_models import get_dashboard_predictor, prepare_product_data


def train_all_models():
    """训练所有模型并保存"""
    print("开始预训练模型...")
    
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
    
    # 训练大型销售数据模型
    if not df_large.empty:
        print("训练大型销售数据模型...")
        large_predictor = get_large_sales_predictor()
        time_series = prepare_time_series_data(df_large)
        if not time_series.empty:
            large_predictor.train(time_series)
    
    # 训练单公司销售数据模型
    if not df_company.empty:
        print("训练单公司销售数据模型...")
        company_predictor = get_company_sales_predictor()
        # 获取销售额最高的公司
        company_overview = df_company.groupby('company_name')['sales_amount'].sum().reset_index()
        top_company = company_overview.sort_values('sales_amount', ascending=False).iloc[0]['company_name'] if not company_overview.empty else None
        
        if top_company:
            time_series = prepare_company_time_series_data(df_company, top_company)
            if not time_series.empty:
                company_predictor.train(time_series)
    
    # # 训练仪表盘产品类型模型
    # if not df_large.empty or not df_company.empty:
    #     print("训练仪表盘产品类型模型...")
    #     dashboard_predictor = get_dashboard_predictor()
    #
    #     # 训练钢化玻璃模型
    #     tempered_data = prepare_product_data(df_large, df_company, '钢化玻璃')
    #     if not tempered_data.empty:
    #         dashboard_predictor.train_product_model(tempered_data, '钢化玻璃')
    #
    #     # 训练夹胶玻璃模型
    #     laminated_data = prepare_product_data(df_large, df_company, '夹层玻璃')
    #     if not laminated_data.empty:
    #         dashboard_predictor.train_product_model(laminated_data, '夹层玻璃')
    #
    print("模型预训练完成！")


# 当直接运行此脚本时，执行模型训练
if __name__ == "__main__":
    train_all_models()