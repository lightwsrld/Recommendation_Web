from django.shortcuts import render, redirect
from .models import User, Predict
from django.http import HttpResponse
from django.contrib.auth.hashers import make_password, check_password
from .forms import LoginForm
from django.http.request import HttpRequest
from joblib import load
import pandas as pd
import user_recomm
import os

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
    if request.method == "POST":
        predict = Predict()
        predict.id = request.POST['id']

        prediction = user_recomm.predict(predict.id)

        dir = "데이터/포스터"
        files = os.listdir(dir)
        matching = [s for s in files if prediction.index[0] in s] 

        img = "/데이터/포스터/" + matching[0]

        context = {
        "id" : predict.id,
        "img" : img,
        "first" : prediction.index[0],
        "prediction" : prediction.to_html()
        }

        return render(request, 'index.html', context=context)
    
    return render(request, 'index.html')