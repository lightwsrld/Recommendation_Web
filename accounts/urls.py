from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from .views import home, predict, predict_1, register, login, logout

app_name = "accounts"

urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict, name='predict'),
    path('predict_1/', predict_1, name='predict_1'),
    path('register/', register, name='register'),
    path('login/', login, name='login'),
    path('logout/', logout, name='logout'),
    path('boards/', include('boards.urls')),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)