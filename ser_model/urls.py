from . import views
from django.urls import path

urlpatterns = [
    path('', views.home, name='home'),
    path('feedback/<slug:bank_slug>/', views.feedback, name='feedback'),
    path("voice_record/<int:bank_id>", views.voice_record, name="voice_record")
    

]
