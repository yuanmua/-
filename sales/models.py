from django.db import models

class SalesData(models.Model):
    """销售数据模型"""
    month = models.CharField('月份', max_length=10)
    product = models.CharField('产品', max_length=50)
    sales_volume = models.FloatField('销售量', null=True, blank=True)
    sales_amount = models.FloatField('销售额')
    cost = models.FloatField('成本', null=True, blank=True)
    gross_margin = models.FloatField('毛利率')
    unit_price = models.FloatField('单价')
    material_ratio = models.FloatField('材料比例')
    major_client = models.CharField('主要客户', max_length=100)
    
    class Meta:
        verbose_name = '销售数据'
        verbose_name_plural = verbose_name
        
    def __str__(self):
        return f"{self.month}-{self.product}"