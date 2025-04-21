import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from django.conf import settings
from django.db.models import Sum

from sales.ml_models.model_utils import load_model, save_model, preprocess_data, build_ensemble_model, dynamic_rolling_predict

# 模型文件路径
MODEL_DIR = os.path.join(settings.BASE_DIR, 'sales', 'ml_models', 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 产品类型模型文件路径
TEMPERED_MODEL_PATH = os.path.join(MODEL_DIR, 'tempered_glass_model.pkl')
LAMINATED_MODEL_PATH = os.path.join(MODEL_DIR, 'laminated_glass_model.pkl')


class DashboardPredictor:
    """仪表盘销售数据预测器"""
    
    def __init__(self):
        self.tempered_model = self._load_or_create_model(TEMPERED_MODEL_PATH)
        self.laminated_model = self._load_or_create_model(LAMINATED_MODEL_PATH)
        
    def _load_or_create_model(self, model_path):
        """加载或创建模型"""
        model = load_model(model_path)
        if model is None:
            model = build_ensemble_model()
            save_model(model, model_path)
        return model
    
    def train_product_model(self, product_data, product_type):
        """训练产品类型模型"""
        # 确保有足够的数据进行预测
        if not product_data.empty and len(product_data) >= 5:  # 至少需要5个数据点
            # 按日期排序
            product_data = product_data.sort_values('sale_date')
            
            # 准备特征
            features = ['month', 'quarter', 'is_month_end', 'lag_1_quantity', 'lag_1_sales', 'rolling_3_quantity', 'rolling_3_sales']
            available_features = [f for f in features if f in product_data.columns]
            
            if len(available_features) >= 3:  # 至少需要3个特征
                X = product_data[available_features]
                y = product_data['sales']
                
                # 使用时间序列交叉验证
                tscv = TimeSeriesSplit(n_splits=2)
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # 选择并训练模型
                    model = self.tempered_model if product_type == '钢化玻璃' else self.laminated_model
                    y_pred = dynamic_rolling_predict(model, X_train, y_train, X_test)
                    
                # 保存训练好的模型
                model_path = TEMPERED_MODEL_PATH if product_type == '钢化玻璃' else LAMINATED_MODEL_PATH
                save_model(model, model_path)
                return True
            else:
                # 如果特征不足，使用简单线性回归
                product_data['month_num'] = product_data['month']
                X = product_data[['month_num']]
                if 'sales' in product_data.columns:
                    y = product_data['sales']
                elif 'sales_amount' in product_data.columns:
                    y = product_data['sales_amount']
                elif 'total_amount' in product_data.columns:
                    y = product_data['total_amount']
                else:
                    return False
                
                model = LinearRegression()
                model.fit(X, y)
                
                # 保存训练好的模型
                model_path = TEMPERED_MODEL_PATH if product_type == '钢化玻璃' else LAMINATED_MODEL_PATH
                save_model(model, model_path)
                return True
        return False
    
    def predict_next_month(self, product_data, product_type):
        """预测下一个月的销售额"""
        # 确保有足够的数据进行预测
        if not product_data.empty and len(product_data) >= 3:  # 至少需要3个数据点
            # 按日期排序
            product_data = product_data.sort_values('sale_date')
            
            # 准备特征
            features = ['month', 'quarter', 'is_month_end', 'lag_1_quantity', 'lag_1_sales', 'rolling_3_quantity', 'rolling_3_sales']
            available_features = [f for f in features if f in product_data.columns]
            
            if len(available_features) >= 3:  # 至少需要3个特征
                # 获取最近的月份
                last_month = product_data['month'].iloc[-1]
                next_month = (last_month % 12) + 1
                
                # 创建预测特征
                next_month_features = product_data.iloc[-1:][available_features].copy()
                next_month_features['month'] = next_month
                next_month_features['quarter'] = (next_month - 1) // 3 + 1
                
                # 选择模型并预测
                model = self.tempered_model if product_type == '钢化玻璃' else self.laminated_model
                predicted_sales = model.predict(next_month_features)[0]
                
                # 计算模型评价指标
                X = product_data[available_features]
                y = product_data['sales']
                y_pred = model.predict(X)
                
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                month_order = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
                
                return {
                    'next_month': month_order[next_month-1],
                    'predicted_sales': round(float(predicted_sales), 2),
                    'model_metrics': {
                        'mse': round(float(mse), 2),
                        'r2': round(float(r2), 2)
                    }
                }
            else:
                # 如果特征不足，使用简单线性回归
                product_data['month_num'] = product_data['month']
                X = product_data[['month_num']]
                if 'sales' in product_data.columns:
                    y = product_data['sales']
                elif 'sales_amount' in product_data.columns:
                    y = product_data['sales_amount']
                elif 'total_amount' in product_data.columns:
                    y = product_data['total_amount']
                else:
                    return None
                
                model = LinearRegression()
                model.fit(X, y)
                
                # 预测下一个月
                next_month = (product_data['month_num'].max() % 12) + 1
                predicted_sales = model.predict([[next_month]])[0]
                
                month_order = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
                
                return {
                    'next_month': month_order[next_month-1],
                    'predicted_sales': round(float(predicted_sales), 2),
                    'model_type': 'linear_regression'
                }
        return None


def get_dashboard_predictor():
    """获取仪表盘预测器实例"""
    return DashboardPredictor()


def prepare_product_data(df_large, df_company, product_type):
    """准备产品类型数据"""
    # 合并相关产品数据
    product_data_large = pd.DataFrame()
    product_data_company = pd.DataFrame()
    
    if not df_large.empty:
        for product in df_large['product_name'].unique():
            if (product_type == '钢化玻璃' and '钢化' in product) or (product_type == '夹胶玻璃' and '夹胶' in product):
                product_data_large = pd.concat([product_data_large, df_large[df_large['product_name'] == product]])
    
    if not df_company.empty:
        for product in df_company['product_name'].unique():
            if (product_type == '钢化玻璃' and '钢化' in product) or (product_type == '夹胶玻璃' and '夹胶' in product):
                product_data_company = pd.concat([product_data_company, df_company[df_company['product_name'] == product]])
    
    # 合并两个数据源
    product_data = pd.DataFrame()
    
    if not product_data_large.empty:
        product_data_large['sales'] = product_data_large['total_amount']
        product_data = pd.concat([product_data, product_data_large])
    
    if not product_data_company.empty:
        product_data_company['sales'] = product_data_company['sales_amount']
        product_data = pd.concat([product_data, product_data_company])
    
    return product_data