from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("about/", views.about, name="about"),
    path("team/", views.contact, name="contact"),
    path("deployment/", views.deployment, name="deployment"),
    path("repellent-control/", views.repellent_control, name="repellent_control"),

    path("register/", views.register, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
]
