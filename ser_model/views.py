from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def welcome(request):
    return render(request, 'index.html')


def feedback(request):
    return render(request, 'feedback.html')