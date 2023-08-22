from . import views
from django.urls import path

urlpatterns = [
    path('', views.welcome, name='Welcome'),
    path('feedback/', views.feedback, name='feedback')
]
