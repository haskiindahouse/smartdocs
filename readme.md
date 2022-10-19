<h1>SmartDocs :boom:</h1>
<h2>About</h3>
The service allows you to semantically compare two versions of a document.
<h2>How it work</h2>
![pipeline](images/pipeline.png)
<h2>Installation :wrench:</h2>

`pip install -r requirements.txt`
<br>
`python -m spacy download ru_core_news_md`<br>
`python manage.py migrate --run-syncdb`<br>
`python manage.py makemigration`<br>
`python manage.py createsuperuser`<br>
`python manage.py runserver`<br>
<h2>Features :shipit: :shipit: :shipit:</h2>
1. Logging<br>
2. Deleting documents from DB<br>
3. Saving the results of processing in the database<br>
4. Integration with JIRA/REDMINE<br>
5. Saving results in convenient formats<br>
6. Selection of important criteria for evaluating documents by the user<br>
7. Manual training of the model through the service<br>
