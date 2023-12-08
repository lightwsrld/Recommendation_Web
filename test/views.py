# views.py
from django.shortcuts import render
from django.http import HttpResponse
from django.http.request import HttpRequest

def my_view(request:HttpRequest, *args, **kwargs):
    result = None
    
    if request.method == 'POST':
        # 폼이 제출되었을 때 폼 데이터 처리

        # result 변수에 어떤 결과를 넣는다고 가정
        result = request.POST['id'] + "님의 폼이 성공적으로 제출되었습니다."

    return render(request, 'test_pt_2.html', {'result': result})