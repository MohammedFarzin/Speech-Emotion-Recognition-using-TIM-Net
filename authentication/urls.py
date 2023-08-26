from . import views
from django.urls import path

urlpatterns = [
    path('signin/', views.signin, name='signin'),
    path('signout', views.signout,name='signout'),

]
