from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("video_feed/", views.video_feed, name="video_feed"),
    path("predict_emotion/", views.predict_emotion, name="predict_emotion"),
    path("predict_video/", views.predict_video, name="predict_video"),
]
