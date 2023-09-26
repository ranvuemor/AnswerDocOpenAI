from django.db import models
import datetime
from django.utils import timezone

# Create your models here.
class PDF(models.Model):

    title = models.CharField(max_length=80)
    pdf = models.FileField(upload_to='pdfs/')
    up_date = models.DateTimeField(auto_now_add=True)