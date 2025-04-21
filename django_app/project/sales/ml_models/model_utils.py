import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from django.conf import settings
from django.db.models import Sum

# 创建模型存储目录
MODEL_DIR = os.path.join(settings.BASE_DIR, 'sales', 'ml_models', 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 模型文件路径
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
LINEAR_MODEL_PATH = os.path.join(MODEL_DIR, 'linear_model.pkl')
CASCADE_MODEL_PATH = os.path.join(MODEL_DIR, 'cascade_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')


def build_ensemble_model():
    """构建集成学习模型"""
    # 基模型配置
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )

    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        subsample=0.8,
        tree_method='hist'
    )

    gbrt = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # 堆叠集成
    return StackingRegressor(
        estimators=[('rf', rf), ('xgb', xgb), ('gbrt', gbrt)],
        final_estimator=RandomForestRegressor(n_estimators=50),
        passthrough=True
    )


def build_cascade_model():
    """构建三级级联结构模型"""
    # 随机森林回归器
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    
    # XGBoost回归器
    xgb_model = XGBRegressor(learning_rate=0.05, n_estimators=200, random_state=42)
    
    # 梯度提升树回归器
    gbdt_model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=200, random_state=42)
    
    # 元模型
    meta_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'gbdt_model': gbdt_model,
        'meta_model': meta_model
    }


def save_model(model, file_path):
    """保存模型到文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到: {file_path}")


def load_model(file_path):
    """从文件加载模型"""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"模型已从 {file_path} 加载")
        return model
    except (FileNotFoundError, EOFError) as e:
        print(f"加载模型时出错: {e}")
        return None


def get_ensemble_model():
    """获取集成模型，如果不存在则创建并保存"""
    model = load_model(ENSEMBLE_MODEL_PATH)
    if model is None:
        model = build_ensemble_model()
        save_model(model, ENSEMBLE_MODEL_PATH)
    return model


def get_linear_model():
    """获取线性回归模型，如果不存在则创建并保存"""
    model = load_model(LINEAR_MODEL_PATH)
    if model is None:
        model = LinearRegression()
        save_model(model, LINEAR_MODEL_PATH)
    return model


def get_cascade_model():
    """获取级联模型，如果不存在则创建并保存"""
    model = load_model(CASCADE_MODEL_PATH)
    if model is None:
        model = build_cascade_model()
        save_model(model, CASCADE_MODEL_PATH)
    return model


def get_or_create_scaler():
    """获取或创建数据标准化器"""
    scaler = load_model(SCALER_PATH)
    if scaler is None:
        scaler = MinMaxScaler()
        save_model(scaler, SCALER_PATH)
    return scaler


def dynamic_rolling_predict(model, X_train, y_train, X_test):
    """动态滚动预测"""
    model.fit(X_train, y_train)
    predictions = []
    X_test = X_test.copy()

    for i in range(len(X_test)):
        # 获取当前步特征
        current_features = X_test.iloc[i:i + 1]

        # 预测并记录
        pred = model.predict(current_features)[0]
        predictions.append(pred)

        # 更新后续lag特征（仅当不是最后一步时）
        if i < len(X_test) - 1 and 'lag_1_sales' in X_test.columns:
            X_test.iloc[i + 1, X_test.columns.get_loc('lag_1_sales')] = pred

    return np.array(predictions)


def preprocess_data(df):
    """数据预处理与特征工程"""
    # 提取月份和年份
    df = df.dropna()
    
    # 转换日期格式
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df.dropna(subset=['sale_date'])
    
    df['month'] = df['sale_date'].dt.month
    df['year'] = df['sale_date'].dt.year
    df['month_str'] = df['month'].apply(lambda x: f"{x}月")
    
    # 创建时间特征
    df['day_of_month'] = df['sale_date'].dt.day
    df['quarter'] = df['sale_date'].dt.quarter
    df['is_month_end'] = df['sale_date'].dt.is_month_end.astype(int)
    
    # 滞后特征 - 按产品和月份分组计算
    for product in df['product_name'].unique():
        product_data = df[df['product_name'] == product].sort_values('sale_date')
        if len(product_data) > 1:
            # 计算销售量的滞后特征
            df.loc[df['product_name'] == product, 'lag_1_quantity'] = product_data['quantity'].shift(1)
            # 计算销售额的滞后特征
            if 'sales_amount' in df.columns:
                df.loc[df['product_name'] == product, 'lag_1_sales'] = product_data['sales_amount'].shift(1)
            elif 'total_amount' in df.columns:
                df.loc[df['product_name'] == product, 'lag_1_sales'] = product_data['total_amount'].shift(1)
    
    # 计算滚动平均
    for product in df['product_name'].unique():
        product_data = df[df['product_name'] == product].sort_values('sale_date')
        if len(product_data) >= 3:
            # 计算销售量的3个月滚动平均
            df.loc[df['product_name'] == product, 'rolling_3_quantity'] = product_data['quantity'].rolling(3).mean().values
            # 计算销售额的3个月滚动平均
            if 'sales_amount' in df.columns:
                df.loc[df['product_name'] == product, 'rolling_3_sales'] = product_data['sales_amount'].rolling(3).mean().values
            elif 'total_amount' in df.columns:
                df.loc[df['product_name'] == product, 'rolling_3_sales'] = product_data['total_amount'].rolling(3).mean().values
    
    # 处理缺失值
    df = df.fillna(0)
    return df