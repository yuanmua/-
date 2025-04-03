from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

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

class Profile(models.Model):
    """用户配置文件模型，扩展Django内置User模型"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    department = models.CharField('部门', max_length=50, blank=True)
    position = models.CharField('职位', max_length=50, blank=True)
    phone = models.CharField('电话', max_length=20, blank=True)
    
    class Meta:
        verbose_name = '用户配置文件'
        verbose_name_plural = verbose_name
        
    def __str__(self):
        return f"{self.user.username}的配置文件"

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """当创建用户时自动创建对应的配置文件"""
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """当保存用户时自动保存对应的配置文件"""
    instance.profile.save()