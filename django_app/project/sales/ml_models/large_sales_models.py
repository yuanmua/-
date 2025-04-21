import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from django.conf import settings

from sales.ml_models.model_utils import load_model, save_model, preprocess_data

# 模型文件路径
MODEL_DIR = os.path.join(settings.BASE_DIR, 'sales', 'ml_models', 'saved_models')
LARGE_SALES_MODEL_PATH = os.path.join(MODEL_DIR, 'large_sales_model.pkl')


class LargeSalesPredictor:
    """大型销售数据预测器"""
    
    def __init__(self):
        self.model = self._load_or_create_model()
        
    def _load_or_create_model(self):
        """加载或创建模型"""
        model = load_model(LARGE_SALES_MODEL_PATH)
        if model is None:
            model = LinearRegression()
            save_model(model, LARGE_SALES_MODEL_PATH)
        return model
    
    def train(self, time_series_data):
        """训练模型"""
        if len(time_series_data) >= 3:  # 至少需要3个数据点
            # 创建时间特征
            time_series_data['time_index'] = time_series_data.index
            
            # 训练线性回归模型
            X = time_series_data[['time_index']]
            y = time_series_data['total_amount']
            self.model.fit(X, y)
            
            # 保存训练好的模型
            save_model(self.model, LARGE_SALES_MODEL_PATH)
            return True
        return False
    
    def predict_future(self, time_series_data, months=3):
        """预测未来几个月的销售额"""
        if len(time_series_data) < 3:
            return None
        
        # 创建时间特征
        time_series_data['time_index'] = time_series_data.index
        
        # 预测未来几个月
        future_indices = np.array(range(len(time_series_data), len(time_series_data) + months)).reshape(-1, 1)
        future_sales = self.model.predict(future_indices)
        
        # 生成未来月份标签
        last_date = pd.to_datetime(f"{int(time_series_data.iloc[-1]['year'])}-{int(time_series_data.iloc[-1]['month'])}-01")
        future_dates = []
        for i in range(1, months + 1):
            next_date = last_date + pd.DateOffset(months=i)
            future_dates.append(f"{next_date.year}-{next_date.month}")
        
        return {
            'dates': future_dates,
            'values': future_sales.tolist()
        }


def get_large_sales_predictor():
    """获取大型销售数据预测器实例"""
    return LargeSalesPredictor()


def prepare_time_series_data(df):
    """准备时间序列数据"""
    # 确保日期格式正确
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['month'] = df['sale_date'].dt.month
        df['year'] = df['sale_date'].dt.year
    
    # 按月份统计销售趋势
    if 'month' in df.columns and 'year' in df.columns:
        time_series = df.groupby(['year', 'month'])['total_amount'].sum().reset_index()
        # 按时间排序并排除最新月份
        time_series = time_series.sort_values(['year', 'month'])
        if not time_series.empty:
            time_series = time_series.iloc[:-1]  # 移除最后一个月
        time_series['date_label'] = time_series.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}", axis=1)
        return time_series
    
    return pd.DataFrame()