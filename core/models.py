from django.contrib.auth.models import User
from django.db import models


class ProcessedVideo(models.Model):
    teacher = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    original_video = models.FileField(upload_to="videos/original/")
    processed_video = models.FileField(upload_to="videos/processed/")
    detected_emotion = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return (
            f"{self.teacher.username} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
        )
