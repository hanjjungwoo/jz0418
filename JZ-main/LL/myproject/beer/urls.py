from django.urls import path
from . import views

urlpatterns = [
    path('ver1', views.ver1, name='ver1'),
    path('ver3', views.ver3, name='ver3'),
    path('hotel', views.hotel, name='hotel'),
]
