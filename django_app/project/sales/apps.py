from django.apps import AppConfig
import os

class SalesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sales'
    
    def ready(self):
        """
        在应用启动时执行初始化操作
        包括预训练模型并保存
        """
        # 避免在开发环境中重复执行
        if os.environ.get('RUN_MAIN', None) != 'true':
            # 导入模型训练模块
            from sales.ml_models.train_models import train_all_models
            
            # 预训练模型
            try:
                train_all_models()
                print("模型预训练完成！")
            except Exception as e:
                print(f"模型预训练失败: {e}")