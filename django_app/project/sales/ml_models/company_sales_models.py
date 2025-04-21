import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from django.conf import settings

from sales.ml_models.model_utils import load_model, save_model, preprocess_data, get_or_create_scaler

# 模型文件路径
MODEL_DIR = os.path.join(settings.BASE_DIR, 'sales', 'ml_models', 'saved_models')
COMPANY_CASCADE_MODEL_PATH = os.path.join(MODEL_DIR, 'company_cascade_model.pkl')


class CompanySalesPredictor:
    """单公司销售数据预测器"""
    
    def __init__(self):
        self.models = self._load_or_create_models()
        self.scaler = get_or_create_scaler()
        
    def _load_or_create_models(self):
        """加载或创建模型"""
        models = load_model(COMPANY_CASCADE_MODEL_PATH)
        if models is None:
            # 创建三级级联结构模型
            models = {
                'rf_model': RandomForestRegressor(n_estimators=200, random_state=42),
                'xgb_model': XGBRegressor(learning_rate=0.05, n_estimators=200, random_state=42),
                'gbdt_model': GradientBoostingRegressor(learning_rate=0.05, n_estimators=200, random_state=42),
                'meta_model': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            }
            save_model(models, COMPANY_CASCADE_MODEL_PATH)
        return models
    
    def train(self, time_series_data):
        """训练模型"""
        if len(time_series_data) >= 3:  # 至少需要3个数据点
            # 创建时间特征
            time_series_data['time_index'] = range(len(time_series_data))
            
            # 数据标准化
            sales_scaled = self.scaler.fit_transform(time_series_data[['sales_amount']])
            time_series_data['sales_amount_scaled'] = sales_scaled
            
            # 准备特征和目标变量
            X = time_series_data[['time_index']]
            y = time_series_data['sales_amount']
            
            # 1. 基模型预测层 - 训练三个基础模型
            self.models['rf_model'].fit(X, y)
            rf_pred = self.models['rf_model'].predict(X)
            
            self.models['xgb_model'].fit(X, y)
            xgb_pred = self.models['xgb_model'].predict(X)
            
            self.models['gbdt_model'].fit(X, y)
            gbdt_pred = self.models['gbdt_model'].predict(X)
            
            # 2. 特征增强层 - 将原始特征与基模型预测结果拼接
            enhanced_features = np.column_stack((X, rf_pred.reshape(-1, 1), 
                                              xgb_pred.reshape(-1, 1), 
                                              gbdt_pred.reshape(-1, 1)))
            
            # 3. 元模型决策层
            self.models['meta_model'].fit(enhanced_features, y)
            
            # 保存训练好的模型
            save_model(self.models, COMPANY_CASCADE_MODEL_PATH)
            save_model(self.scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
            return True
        return False
    
    def predict_future(self, time_series_data, months=3):
        """预测未来几个月的销售额"""
        if len(time_series_data) < 3:
            return None
        
        # 创建时间特征
        time_series_data['time_index'] = range(len(time_series_data))
        
        # 准备特征
        X = time_series_data[['time_index']]
        
        # 预测未来几个月
        future_indices = np.array(range(len(time_series_data), len(time_series_data) + months)).reshape(-1, 1)
        
        # 使用基模型进行预测
        rf_future = self.models['rf_model'].predict(future_indices)
        xgb_future = self.models['xgb_model'].predict(future_indices)
        gbdt_future = self.models['gbdt_model'].predict(future_indices)
        
        # 构建增强特征
        future_enhanced = np.column_stack((future_indices, rf_future.reshape(-1, 1), 
                                        xgb_future.reshape(-1, 1), gbdt_future.reshape(-1, 1)))
        
        # 使用元模型进行最终预测
        future_sales = self.models['meta_model'].predict(future_enhanced)
        
        # 生成未来月份标签
        last_date = pd.to_datetime(f"{int(time_series_data.iloc[-1]['year'])}-{int(time_series_data.iloc[-1]['month'])}-01")
        future_dates = []
        for i in range(1, months + 1):
            next_date = last_date + pd.DateOffset(months=i)
            future_dates.append(f"{next_date.year}-{next_date.month}")
        
        # 计算模型评价指标
        y = time_series_data['sales_amount']
        y_pred = self._predict_with_trained_models(X)
        
        # 确保数据为numpy数组且维度正确
        y = np.array(y, dtype=float).flatten()
        y_pred = np.array(y_pred, dtype=float).flatten()
        
        # 计算评价指标
        metrics = self._calculate_metrics(y, y_pred)
        
        return {
            'dates': future_dates,
            'values': future_sales.tolist(),
            'metrics': metrics
        }
    
    def _predict_with_trained_models(self, X):
        """使用训练好的模型进行预测"""
        rf_pred = self.models['rf_model'].predict(X)
        xgb_pred = self.models['xgb_model'].predict(X)
        gbdt_pred = self.models['gbdt_model'].predict(X)
        
        enhanced_features = np.column_stack((X, rf_pred.reshape(-1, 1), 
                                          xgb_pred.reshape(-1, 1), 
                                          gbdt_pred.reshape(-1, 1)))
        
        return self.models['meta_model'].predict(enhanced_features)
    
    def _calculate_metrics(self, y, y_pred):
        """计算模型评价指标"""
        try:
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            # 计算MAPE，避免除以零
            mask = y != 0
            y_masked = y[mask]
            y_pred_masked = y_pred[mask]
            mape = np.mean(np.abs((y_masked - y_pred_masked) / y_masked)) * 100 if len(y_masked) > 0 else 0
            
            # 计算相关系数
            if len(y) != len(y_pred) or len(y) < 2:
                corr = 0.0
            else:
                corr_matrix = np.corrcoef(y, y_pred)
                corr = corr_matrix[0, 1] if corr_matrix.size >= 4 else 0.0
                
            return {
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'corr': float(corr)
            }
        except Exception as e:
            print(f"计算评价指标时出错: {str(e)}")
            return {
                'rmse': 0.0,
                'mae': 0.0,
                'mape': 0.0,
                'corr': 0.0
            }


def get_company_sales_predictor():
    """获取单公司销售数据预测器实例"""
    return CompanySalesPredictor()


def prepare_company_time_series_data(df, company_name):
    """准备单公司时间序列数据"""
    # 筛选出指定公司的数据
    company_data = df[df['company_name'] == company_name]
    
    # 确保日期格式正确
    if 'sale_date' in company_data.columns:
        company_data['sale_date'] = pd.to_datetime(company_data['sale_date'])
        company_data['month'] = company_data['sale_date'].dt.month
        company_data['year'] = company_data['sale_date'].dt.year
    
    # 按月份统计销售趋势
    if 'month' in company_data.columns and 'year' in company_data.columns:
        time_series = company_data.groupby(['year', 'month'])['sales_amount'].sum().reset_index()
        # 按时间排序并排除最新月份
        time_series = time_series.sort_values(['year', 'month'])
        if not time_series.empty:
            time_series = time_series.iloc[:-1]  # 移除最后一个月
        time_series['date_label'] = time_series.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}", axis=1)
        return time_series
    
    return pd.DataFrame()