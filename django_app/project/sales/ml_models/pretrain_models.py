import os
import pandas as pd
import numpy as np
from django.conf import settings
from django.db.models import Sum

from sales.models.models_company_sales import GlassProcessingSalesSingleCompany
from sales.models.models_large_sales import GlassProcessingSalesLarge
from sales.ml_models.model_utils import preprocess_data
from sales.ml_models.dashboard_models import get_dashboard_predictor, prepare_product_data

# 模型文件路径
MODEL_DIR = os.path.join(settings.BASE_DIR, 'sales', 'ml_models', 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)


def pretrain_dashboard_models():
    """预训练仪表盘模型并保存"""
    print("开始预训练仪表盘模型...")
    
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
    
    # 训练仪表盘产品类型模型
    if not df_large.empty or not df_company.empty:
        print("训练仪表盘产品类型模型...")
        dashboard_predictor = get_dashboard_predictor()
        
        # 训练钢化玻璃模型
        tempered_data = prepare_product_data(df_large, df_company, '钢化玻璃')
        if not tempered_data.empty and len(tempered_data) >= 3:  # 至少需要3个数据点
            print("训练钢化玻璃模型...")
            dashboard_predictor.train_product_model(tempered_data, '钢化玻璃')
        
        # 训练夹胶玻璃模型
        laminated_data = prepare_product_data(df_large, df_company, '夹胶玻璃')
        if not laminated_data.empty and len(laminated_data) >= 3:  # 至少需要3个数据点
            print("训练夹胶玻璃模型...")
            dashboard_predictor.train_product_model(laminated_data, '夹胶玻璃')
    
    print("仪表盘模型预训练完成！")


# 当直接运行此脚本时，执行模型训练
if __name__ == "__main__":
    pretrain_dashboard_models()