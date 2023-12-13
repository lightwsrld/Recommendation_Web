from django.shortcuts import render,redirect,get_object_or_404
from .models import Board, Predict, News21_1, News21_2, News21_3, News21_4, News22_1, News22_2, News22_3, News22_4
from accounts.models import User
# from .models import Tag
#from tag.models import Tag
from django.http.request import HttpRequest
import pandas as pd
from joblib import load
from keras.layers import *
from keras.models import *
from keras import backend as K
import csv


def board_main(request):
    redirect('/')

def test(request):
    return render(request, 'test.html')

def apart(request):
    return render(request, 'apart.html')

def map3(request):
    return render(request, 'HB.html')

def map4(request):
    return render(request, 'New.html')

def 부동산실거래가(request):
    return render(request, 'news/news21_1.html')

def lda_21_1(request):
    return render(request, 'lda/lda_21_1.html') 

def lda_21_2(request):
    return render(request, 'lda/lda_21_2.html') 

def lda_21_3(request):
    return render(request, 'lda/lda_21_3.html') 

def lda_21_4(request):
    return render(request, 'lda/lda_21_4.html') 

def lda_21_5(request):
    return render(request, 'lda/lda_21_5.html') 

def lda_21_6(request):
    return render(request, 'lda/lda_21_6.html')    

def lda_21_7(request):
    return render(request, 'lda/lda_21_7.html') 

def lda_21_8(request):
    return render(request, 'lda/lda_21_8.html') 

def lda_21_9(request):
    return render(request, 'lda/lda_21_9.html') 

def lda_21_10(request):
    return render(request, 'lda/lda_21_10.html') 

def lda_21_11(request):
    return render(request, 'lda/lda_21_11.html') 

def lda_21_12(request):
    return render(request, 'lda/lda_21_12.html') 

def lda_22_1(request):
    return render(request, 'lda/lda_22_1.html') 

def lda_22_2(request):
    return render(request, 'lda/lda_22_2.html')

def lda_22_3(request):
    return render(request, 'lda/lda_22_3.html')

def lda_22_4(request):
    return render(request, 'lda/lda_22_4.html')

def lda_22_5(request):
    return render(request, 'lda/lda_22_5.html')

def lda_22_6(request):
    return render(request, 'lda/lda_22_6.html')

def lda_22_7(request):
    return render(request, 'lda/lda_22_7.html')

def lda_22_8(request):
    return render(request, 'lda/lda_22_8.html')

def lda_22_9(request):
    return render(request, 'lda/lda_22_9.html')

def lda_22_10(request):
    return render(request, 'lda/lda_22_10.html')

def lda_22_11(request):
    return render(request, 'lda/lda_22_11.html')

def lda_22_12(request):
    return render(request, 'lda/lda_22_12.html')           


def predict(request:HttpRequest, *args, **kwargs):
    if request.method == "POST":
        predict = Predict()
        predict.first = request.POST['first']
        predict.second = request.POST['second']
        predict.third = request.POST['third']
        predict.fourth = request.POST['fourth']

        model = load('boards\house.joblib')

        input_variables = pd.DataFrame([[predict.first, predict.second, predict.third, predict.fourth]],
                        columns=['금리', '국내총생산', '대출금리', '매매 거래량'],
                        dtype='float',
                        index=['input'])

        prediction = model.predict(input_variables)[0]

        context = {
        "first" : predict.first,
        "second" : predict.second,
        "third" : predict.third,
        "fourth" : predict.fourth,
        "prediction" : prediction
        }

        return render(request, 'predict.html', context=context)
    
    return render(request, 'predict.html')

def news(request):
    

    return render(request, 'news.html')

def news21_1(request):
    class_object = News21_1.objects.all()

    return render(request, 'news/news21_1.html', {'class_object':class_object})

def news21_2(request):
    class_object = News21_2.objects.all()
    return render(request, 'news/news21_2.html', {'class_object':class_object})

def news21_3(request):
    class_object = News21_3.objects.all()
    return render(request, 'news/news21_3.html', {'class_object':class_object})

def news21_4(request):
    class_object = News21_4.objects.all()
    return render(request, 'news/news21_4.html', {'class_object':class_object})

def news22_1(request):
    class_object = News22_1.objects.all()
    return render(request, 'news/news22_1.html', {'class_object':class_object})

def news22_2(request):
    class_object = News22_2.objects.all()
    return render(request, 'news/news22_2.html', {'class_object':class_object})

def news22_3(request):
    class_object = News22_3.objects.all()
    return render(request, 'news/news22_3.html', {'class_object':class_object})

def news22_4(request):
    class_object = News22_4.objects.all()
    return render(request, 'news/news22_4.html', {'class_object':class_object})