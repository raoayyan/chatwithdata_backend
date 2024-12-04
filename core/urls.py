from django.urls import path
from . import views

urlpatterns = [
    path('databases/', views.list_databases),
    path('update-explanation/<str:explanation_id>/', views.update_explanation, name='update_explanation'),
    path('fetch-explanations/<str:db_name>/', views.fetch_explanations, name='fetch_explanations'),
    path('add-database/', views.add_database, name='add_database'),
    path("chat_with_database/", views.chat_with_database, name="chat_with_database"),
]
