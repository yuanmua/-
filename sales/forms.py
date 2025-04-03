from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import Profile

class UserLoginForm(AuthenticationForm):
    """用户登录表单"""
    username = forms.CharField(label='用户名', 
                               widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入用户名'}))
    password = forms.CharField(label='密码', 
                               widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请输入密码'}))
    
    class Meta:
        model = User
        fields = ['username', 'password']

class UserRegisterForm(UserCreationForm):
    """用户注册表单"""
    username = forms.CharField(label='用户名', 
                               widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入用户名'}))
    # email = forms.EmailField(label='邮箱',
    #                          widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': '请输入邮箱'}))
    password1 = forms.CharField(label='密码', 
                                widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请输入密码'}))
    password2 = forms.CharField(label='确认密码', 
                                widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请再次输入密码'}))
    
    class Meta:
        model = User
        fields = ['username', 'password1', 'password2']

        # fields = ['username', 'email', 'password1', 'password2']
    # 可选：完全跳过验证逻辑
    # def _post_clean(self):
    #     # 跳过父类 UserCreationForm 的密码验证
    #     super(forms.ModelForm, self)._post_clean()


class ProfileForm(forms.ModelForm):
    """用户配置文件表单"""
    department = forms.CharField(label='部门', required=False,
                                 widget=forms.TextInput(attrs={'class': 'form-control'}))
    position = forms.CharField(label='职位', required=False,
                               widget=forms.TextInput(attrs={'class': 'form-control'}))
    phone = forms.CharField(label='电话', required=False,
                            widget=forms.TextInput(attrs={'class': 'form-control'}))
    
    class Meta:
        model = Profile
        fields = ['department', 'position', 'phone']

class UserForm(forms.ModelForm):
    """用户基本信息表单"""
    first_name = forms.CharField(label='名', required=False,
                                 widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(label='姓', required=False,
                                widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(label='邮箱',
                             widget=forms.EmailInput(attrs={'class': 'form-control'}))
    
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']