from django import forms
from .models import PDF

class UploadPDFform(forms.ModelForm):
    class Meta:
        model = PDF
        fields = ('title', 'pdf',)