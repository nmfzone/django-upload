# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from django.db import models


class File(models.Model):
    location = models.FileField(upload_to='upload')
