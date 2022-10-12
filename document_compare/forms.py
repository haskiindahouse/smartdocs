# -*- coding: utf-8 -*-
from django import forms


class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='Выберите документ',
        help_text=''
    )