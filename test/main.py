import flask
from flask import Flask, request, render_template
from django.http import HttpResponse
import joblib
import pandas as pd
import user_recomm
   
app = Flask(__name__)
   
# index 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')
   
# 이미지 업로드에 대한 예측값 반환
@app.route('/predict', methods=['POST'])
def make_prediction(request:HttpRequest, *args, **kwargs):
    if request.method == 'POST':
   
        # 업로드 파일 처리 분기
        id = request.form['first']
        print(id)

        if not id: return render_template('index.html', context="No ID")
        
        prediction = user_recomm.predict(id)
        print(prediction)

        context = {'prediction' : prediction}

        # 결과 리턴
        return render_template('index.html', context=context)
   
# 미리 학습시켜서 만들어둔 모델 로드
if __name__ == '__main__':
    #model = joblib.load('./model_test_id.pkl')
    app.run(host='0.0.0.0', port=8000, debug=True)