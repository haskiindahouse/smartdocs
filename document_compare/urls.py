from django.urls import path

from . import views

urlpatterns = [
    path("home/", views.index, name='home'),
    path("start/", views.start, name='start'),
    path("compare/", views.compare, name='compare')
]