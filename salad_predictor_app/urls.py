# salad_predictor_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_salad_taste, name='predict_salad_taste'),
]