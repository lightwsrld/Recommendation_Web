from django.shortcuts import render, redirect
from .models import User, Predict
from django.http import HttpResponse
from django.contrib.auth.hashers import make_password, check_password
from .forms import LoginForm
from django.http.request import HttpRequest
from joblib import load
import pandas as pd
from accounts import user_recomm, title_recomm
import unicodedata
from django.conf import settings

def home(request):
    user_id = request.session.get('user')
    if user_id:
        user = User.objects.get(pk = user_id)
        return HttpResponse("Hello! %s님" % user)
    else:
        return render(request, 'index.html')


def register(request):
    if request.method == 'GET':
        return render(request, 'register.html')

    elif request.method == 'POST':
        username = request.POST.get('username', None)
        useremail = request.POST.get('useremail', None) 
        password = request.POST.get('password', None)
        re_password = request.POST.get('re_password', None)
        
        err_data={}
        if not(username and useremail and password and re_password):
            err_data['error'] = '모든 값을 입력해주세요.'
            return render(request, 'register.html', err_data)
        
        elif password != re_password:
            err_data['error'] = '비밀번호가 다릅니다.'
            return render(request, 'register.html', err_data)
        
        else:
            user = User(
                username=username,
                useremail=useremail,
                password=make_password(password),
            )
            user.save()
            return redirect('/')

def login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            request.session['user'] = form.user_id
            return redirect('/')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

def logout(request):
    if request.session.get('user'):
        del(request.session['user'])
    return redirect('/')


def predict(request:HttpRequest, *args, **kwargs):
    df = pd.read_csv('데이터\연극+뮤지컬_TOP77.csv')
    df['Image'] = [unicodedata.normalize('NFC', filename) for filename in df['Image']]
    if request.method == "POST":
        predict = Predict()
        predict.id = request.POST['id']

        prediction = user_recomm.predict(predict.id)

        context = {
            "id": predict.id,
            "first" : prediction.index[0]
        }

        first_row = prediction.iloc[0]

        context["img"] = first_row['Image']
        # input_title에 해당하는 이미지 파일 
        context["prediction"] = prediction.to_html()


        return render(request, 'index.html', context=context)
    
    return render(request, 'index.html')

def predict_1(request: HttpRequest, *args, **kwargs):
    df = pd.read_csv('데이터\연극+뮤지컬_TOP77.csv')
    df['Image'] = [unicodedata.normalize('NFC', filename) for filename in df['Image']]

    if request.method == "POST":
        predict_1 = Predict()
        predict_1.title = request.POST['title']

        prediction_1 = title_recomm.image_plus(predict_1.title)

        context = {
            "title": predict_1.title,
        }

        first_row = prediction_1.iloc[0]

        context["first_title"] = first_row['Title']
        context["img"] = first_row['Image']
        # input_title에 해당하는 이미지 파일 
        matching_row = df[df['Title'] == predict_1.title].iloc[0]
        context["matching_img"] = matching_row['Image']
        context["prediction_1"] = prediction_1.to_html()

        return render(request, 'index.html', context=context)

    return render(request, 'index.html')