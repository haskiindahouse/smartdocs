from django.shortcuts import redirect, render
from .models import Document, Sentence
from .forms import DocumentForm
from text_analyzer.text_differ import get_all_text, get_match, get_minus_and_plus, get_json
import json

IMPORTANCE_COLOR = {
    0: "black",
    1: "#dbfa41",
    2: "#face41",
    3: "#faaa41",
    4: "#fa7d41",
    5: "#ff0000"
}

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
        difference, deleted_sentencies, analytics = get_json(t1, t2, d_eq, d_changed, deleted)  # формирование файла разметки
        form = DocumentForm()

        diffs = []
        added = []
        changed = []
        deleted = []

        changed_entities = []

        for diff in difference['eq_and_match']:
            if diff.get('sim_score') is not None:
                changed.append(diff)

        for diff in sorted(difference['eq_and_match'], key=lambda x: x['importance'], reverse=True):
            if diff.get('score') is not None and diff['score'] == 0:
                added.append('<span style="font-size: 35px; color:' + IMPORTANCE_COLOR[diff['importance']] + ';">&#8226;</span>   ' + f'[{diff["id"]}] ' + diff['markdown_ent'].replace('\\', '').replace('\n', ''))
                t2[diff['id']] = ('<span style="background-color: green;">' + t2[diff['id']][0] + '</span>', t2[diff['id']][1])

        for change in sorted(changed, key=lambda x: x['importance'], reverse=True):
            diffs.append('<span style="font-size: 35px; color:' + IMPORTANCE_COLOR[change['importance']] + ';">&#8226;</span>   ' + f'[{change["id"]}] ' + change['markdown'].replace('\\', '').replace('\n', ''))
            changed_entities.append([ '<span style="font-size: 35px; color:' + IMPORTANCE_COLOR[change['importance']] + ';">&#8226;</span>   ' + f'[{change["id"]}][Было] ' + change['markdown_ent_1'].replace('\\', '').replace('\n', ''),
                                      '<span style="font-size: 35px; color:' + IMPORTANCE_COLOR[change['importance']] + ';">&#8226;</span>   ' + f'[{change["id"]}][Стало] ' + change['markdown_ent_2'].replace('\\', '').replace('\n', '')])

        for diff in sorted(difference['deleted'], key=lambda x: x['importance'], reverse=True):
            deleted.append('<span style="font-size: 35px; color:' + IMPORTANCE_COLOR[diff['importance']] + ';">&#8226;</span>   ' + f'[{diff["id"]}] ' +diff['markdown_ent'].replace('\\', '').replace('\n', ''))
            t1[diff['id']] = ('<span style="background-color: red;">' + t1[diff['id']][0] + '</span>', t1[diff['id']][1])


        for key, value in t1.items():
            firstDoc += value[0]
            if value[1]:
                firstDoc += '\n'

        for key, value in t2.items():
            secondDoc += value[0]
            if value[1]:
                secondDoc += '\n'

    documents = Document.objects.all()
    context = {'documents': documents, 'form': form, 'message': message, 'diffs': diffs, 'added': added, 'deleted': deleted,
               'changed_entities': changed_entities, 'deleted_sentencies':deleted_sentencies, 'firstDoc': firstDoc, 'secondDoc': secondDoc
               }

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