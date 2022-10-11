# -*- coding: utf-8 -*-
from django.db import models
from .validators import validate_file_extension


class Document(models.Model):
    docfile = models.FileField(upload_to='documents/%Y/%m/%d', validators=[validate_file_extension])

    @classmethod
    def check(cls, **kwargs):
        return validate_file_extension(cls)

    @property
    def filename(self):
        return self.audio_file.path # os.path.basename(self.audio_file.path)

class Sentence(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    text = models.TextField()
    num_paragraph = models.IntegerField()
    score = models.FloatField()
    n_matches = models.IntegerField()
    entities = models.JSONField()
    date = models.DateField()
