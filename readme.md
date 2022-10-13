<h1>Инструкция по развертыванию</h1>
`pip install -r requirements.txt`
`python -m spacy download ru_core_news_md`
`python manage.py migrate --run-syncdb`<br>
`python manage.py makemigration`<br>
`python manage.py createsuperuser`<br>
`python manage.py runserver`<br>
