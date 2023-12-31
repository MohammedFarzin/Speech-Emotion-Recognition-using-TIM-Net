from django.db import models
from django.urls import reverse
from autoslug import AutoSlugField
from django.contrib.auth.models import User
from django.db.models import Avg, Count

class BankDetails(models.Model):
    bank_name = models.CharField(max_length=200, unique=True)
    slug = AutoSlugField(populate_from='bank_name', unique=True, null=True, default=None)
    description = models.TextField(max_length=500, unique=True)
    # feedbacks = models.ManyToManyField(User, through='Feedback', related_name='bank_feedbacks')

    class Meta:
        verbose_name = 'bank detail'
        verbose_name_plural = 'bank details'

    def get_url(self):
        return reverse('feedback', args=[self.slug])
    

    def average_emotion(self):
        emotion_values = {
            'happy': 100,    
            'sad': 25,
            'neutral': 50,
            'angry': 0,
            'calm' : 75,
            'disgust' : 30,
            'fear' : 30,
            'surprise' : 90
            
        }
        try:
            feedbacks_for_bank = Feedback.objects.filter(bank=self)
            total_emotion_value = sum(emotion_values[feedback.emotion] for feedback in feedbacks_for_bank)
            avg_emotion_percentage = total_emotion_value / len(feedbacks_for_bank)
        except ZeroDivisionError:
             avg_emotion_percentage = 0
        return avg_emotion_percentage
    
    def emotion_count(self):
        feedback_count = Feedback.objects.filter(bank=self).aggregate(count=Count('id'))
        print(feedback_count)
        count = 0
        if feedback_count['count'] is not None:
            count = int(feedback_count['count'])
        return count
        
        

    def __str__(self):
        return self.bank_name

class Feedback(models.Model):
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('neutral', 'Neutral'),
        ('angry', 'Angry'),
        ('calm', 'Calm'),
        ('disgust', 'Disgust'),
        ('fear', 'Fear'),
        ('surprise', 'Surprise')
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedbacks')
    bank = models.ForeignKey(BankDetails, on_delete=models.CASCADE, related_name='bank_feedbacks')
    emotion = models.CharField(max_length=10, choices=EMOTION_CHOICES, default='neutral')


    
    
    def __str__(self):
        return f"{self.user.username} - {self.bank.bank_name}"
    
    

