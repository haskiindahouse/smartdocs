from django.shortcuts import redirect, render
from .models import Document, Sentence
from .forms import DocumentForm


def index(request):
    context = {}
    return render(request, 'index.html', context)


def start(request):
    message = 'Upload as many files as you want!'
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
    if request.method == 'GET':
        form = DocumentForm()
        message = "HELLO WORLD"

    documents = Document.objects.all()

    # Сюда добавить сбор со сервера предложений по документу
    context = {'documents': documents, 'form': form, 'message': message}
    return render(request, 'start.html', context)