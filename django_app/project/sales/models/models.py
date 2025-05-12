from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class SalesData(models.Model):
    """玻璃加工销售数据模型"""
    product_type_choices = (
        ('钢化玻璃', '钢化玻璃'),
        ('夹层玻璃', '夹层玻璃'),
    )
    month = models.CharField('月份', max_length=10)
    product_type = models.CharField('产品类型', max_length=50, choices=product_type_choices)
    sales_volume = models.FloatField('销售量(㎡)')
    sales_amount = models.FloatField('销售额')
    material_ratio = models.FloatField('材料比例')
    forecast_accuracy = models.FloatField('预测准确率', null=True, blank=True)
    timestamp = models.DateTimeField('记录时间', auto_now_add=True)

    class Meta:
        verbose_name = '玻璃销售数据'
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['product_type', 'month']),
        ]

    def __str__(self):
        return f"{self.month}-{self.product_type}"

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