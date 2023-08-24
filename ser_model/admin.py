from django.contrib import admin
from .models import *
# Register your models here.

@admin.register(BankDetails)
class BankDetailsAdmin(admin.ModelAdmin):
    list_display = ('bank_name', 'slug', 'description')

admin.site.register(Feedback)