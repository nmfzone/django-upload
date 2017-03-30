# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from django import forms
from app.models import File


class FileForm(forms.ModelForm):
    class Meta:
        model = File
        fields = ('location', )
