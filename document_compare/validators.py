import os
from django.core.exceptions import ValidationError


def validate_file_extension(value):
  ext = os.path.splitext(str(value.filename))[1]
  print(ext)
  valid_extensions = ['.pdf', '.doc', '.docx']
  return True