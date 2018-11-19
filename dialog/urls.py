from django.contrib import admin
from django.conf.urls import url, include
from . import views
app_name = 'IqiyiProject'
urlpatterns = [
    url('^dialog/$', views.edit_action, name='dialog'),
    url('^dialog/$', views.init_action, name='init')
]