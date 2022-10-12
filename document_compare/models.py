# -*- coding: utf-8 -*-
from django.db import models
from django.core.validators import FileExtensionValidator


class Document(models.Model):
    docfile = models.FileField(null=True, blank=True, upload_to='documents/%Y/%m/%d', validators=[FileExtensionValidator(['doc, docx, rtf'])])


class Sentence(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    text = models.TextField()
    num_paragraph = models.IntegerField()
    score = models.FloatField()
    n_matches = models.IntegerField()
    entities = models.JSONField()
    date = models.DateField()
