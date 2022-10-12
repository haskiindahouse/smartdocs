from django.shortcuts import redirect, render
from .models import Document, Sentence
from .forms import DocumentForm
from text_analyzer.text_differ import get_all_text, get_match, get_minus_and_plus, get_json
import json

def index(request):
    context = {}
    return render(request, 'index.html', context)


def compare(request):
    difference = ""
    message = 'Upload as many files as you want!'

    deleted_sentencies = ""
    firstDoc = ""
    secondDoc = ""
    if request.method == 'POST':
        t1 = get_all_text(request.POST.getlist("doc_check")[0])  # текст 1 путь к файлу (doc,docx,rtf) STRING
        t2 = get_all_text(request.POST.getlist("doc_check")[1])  # текст 2 путь к файлу (doc,docx,rtf) STRINGv
        d_eq, d_changed = get_match(t1, t2)  # словарь полных совпадений и изменений {text1_id: text2_id}
        deleted, added = get_minus_and_plus(t1, t2, d_eq, d_changed)  # удаленные из 1 текста и добавленные во 2 текст
        difference, deleted_sentencies = get_json(t1, t2, d_eq, d_changed, deleted)  # формирование файла разметки
        form = DocumentForm()

        print(type(difference))
        # for diff in difference['eq_and_match']:
        #     if diff.get('score') is not None and diff['score'] == 0:
        #         t2[diff['id']] = f'<span style="background-color: green; "> {t2[diff["id"]][0]}</span>'

        for key, value in t1.items():
            firstDoc += f'<span style="background-color: green; "> {value[0]}</span>'
            if value[1]:
                firstDoc += '\n'

        for key, value in t2.items():
            secondDoc += value[0]
            if value[1]:
                secondDoc += '\n'

    documents = Document.objects.all()
    context = {'documents': documents, 'form': form, 'message': message, 'deleted_sentencies':deleted_sentencies, 'firstDoc': firstDoc, 'secondDoc': secondDoc}

    return render(request, 'start.html', context)


def start(request):
    difference = ""
    message = 'Upload as many files as you want!'
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
    if request.method == 'GET':
        form = DocumentForm()

    documents = Document.objects.all()
    # Сюда добавить сбор со сервера предложений по документу
    context = {'documents': documents, 'form': form, 'message': message, 'difference': difference, 'deleted_sentencies':"", 'firstDoc': '', 'secondDoc': ''}
    return render(request, 'start.html', context)