from django.urls import path
from . import views

urlpatterns = [
    path('databases/', views.list_databases),
    path('update-explanation/<str:explanation_id>/', views.update_explanation, name='update_explanation'),
    path('fetch-explanations', views.fetch_explanations, name='fetch-explanations'),
    path('add-database/', views.add_database, name='add_database'),
    path("chat_with_database/", views.chat_with_database, name="chat_with_database"),
    path('store-chat/', views.store_chat_view, name='store_chat'),
    path('signup/', views.signup_view, name='signup'),
    path('signin/', views.sign_in_view, name='signin'),

]
