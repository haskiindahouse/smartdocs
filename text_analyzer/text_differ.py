import os
import pathlib
import re
import json
import numpy as np
from random import randint
from collections import Counter
import math
import statistics

from razdel import tokenize, sentenize
from natasha import *

from docx import Document
from striprtf.striprtf import rtf_to_text

# import aspose.words as aw
import difflib as dl
import diff_match_patch as dmp_module
import spacy
from spacy import displacy

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# скачивание модели для определения сущностей  BERT (≈ 1.5 GB)
tokenizer = AutoTokenizer.from_pretrained("surdan/LaBSE_ner_nerel")
model = AutoModelForTokenClassification.from_pretrained("surdan/LaBSE_ner_nerel")
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="first")
model_sim = SentenceTransformer('uaritm/multilingual_en_ru_uk')
nlp_spacy = spacy.load('ru_core_news_md')

# настройка весов сущностей
label_impotance={
    "MONEY" : 1,
    "ORGANIZATION" : 0.5,
    "PERSON" : 0.5,
    "FACILITY" : 0.5
}

label_impotance_alone={
    "MONEY" : 5,
    "ORGANIZATION" : 3,
    "PERSON" : 4,
    "FACILITY" : 2,
    "DATE" : 5,
    "PRODUCT": 4,
    "CITY": 3
}

def get_all_text(filename: str) -> dict:
    '''
    считывание документа и разбиение его на предложения
    с использованием Natasha
    возвращает словарь с номерами строк и их содержанием
    '''
    full_txt = {}
    line_number = 0
    if pathlib.Path(filename).suffix == '.docx':
        doc = Document(filename)
        for i,para in enumerate(doc.paragraphs):
            for sen in list(sentenize(para.text)):
                full_txt[line_number] = (sen.text,i)
                line_number+=1
    else: # работа с форматом rtf
        with open(filename, 'r') as file:
            text = file.read()

        rtf = text
        text1 = rtf_to_text(rtf)

        for sen in list(sentenize(text1)):
            full_txt[line_number] = sen.text
            line_number+=1

    return {key: val for key,val in full_txt.items() if val[0] != ''}


def get_sent_similarity_quickly(query: str, passage: list):
    '''
    вычислений меры "похожести" предложений
    '''
    scores = []
    for i,(sen,_) in passage:
        scores.append(dl.SequenceMatcher(lambda x: x == " ",query,sen).ratio())
    if max(scores) > 0.65:
        return passage[np.argmax(np.asarray(scores), axis=0)][0]
    return False


def get_match(t1: list, t2: list):
    '''
    соотнесение измененных и одинаковых на основе
    прямого сравнивания и меры похожести, основанной
    на косинусном расстоянии между векторами эмбедингов
    '''
    t1 = t1.copy()
    t2 = t2.copy()
    d_eq = {}
    d_changed = {}
    scrap = '',' ','-','.','  ',' .'
    for raw1_k,raw1_v in t1.items():
        not_found = True
        for raw2_k, raw2_v in t2.items():
            if raw1_v[0]==raw2_v[0] and raw2_v[0]!='' and raw2_v[0]!='':
                d_eq[raw2_k] = raw1_k
                del t2[raw2_k]
                not_found = False
                break

        if not_found and raw1_v[0] not in scrap:
            l_values = list(t2.items())
            sim = get_sent_similarity_quickly(raw1_v[0],l_values)
            if sim:
                d_changed[raw1_k] = sim

    return d_eq, dict((v,k) for k,v in d_changed.items())
    # t1 - values | t2 - keys


def get_minus_and_plus(t1,t2,d_eq,d_changed):
    '''
    вычисление пересечений
    на выходе удаленные предложения из 1 текста
    и добавленные во 2-й
    '''
    deleted = set(t1.keys())-set.union(set(d_eq.values()), set(d_changed.values())) # удалено из t1
    added = set(t2)-set.union(set(d_eq.keys()), set(d_changed.keys())) # добавлено в t2
    return deleted, added


def get_sentance_diff(sen1: str, sen2: str) -> list:
    '''
    принимает на вход 2 строки (предложения), вычисляется разница
    на выходе получается список (-1, 0, 19, 'Алгоритм Дойче-Йозе')
    где [0]:
    "-" - измменение или удаление части текста
    "0" - отсутствие изменений
    "1" - добавление нового фрагмента
    [1]:[2] соответствующий фрагмент текста
    [3] текст (сущность)
    '''
    dmp = dmp_module.diff_match_patch()
    diff = dmp.diff_main(sen1, sen2)
    dmp.diff_cleanupSemantic(diff)
    counter = 0
    out = []
    sent = ''
    for el in diff:
        out.append((el[0],counter,counter+len(el[1]),el[1]))
        counter+=len(el[1])
        sennt+=el[1]
    return out, sent


def get_diff_to_html(sen1, sen2):
    '''
    создание HTML размметки по предложениям
    на основе внесенных изменений
    '''
    dmp = dmp_module.diff_match_patch()
    diff = dmp.diff_main(sen1, sen2)
    dmp.diff_cleanupSemantic(diff)

    return dmp.diff_prettyHtml(diff)


def get_sent_similarity(query, passage):
    '''
    вычисление коэффициента "похожести" предложений используя similarity BERT
    принимает строку основного предложения и список из предложений или строку
    возвращает коэф. похожести на основное предложение
    '''
    query_embedding = model_sim.encode(query)
    passage_embedding = model_sim.encode(passage)
    score = util.cos_sim(query_embedding, passage_embedding)

    return score


def get_tag_diff_score(doc,doc1,label_impotance={}):
  '''
  Принимает словари из get_ner_tokens() для 2-х сматченных предложений
  Возращает оценку, основанную на пересечении токенов
  Можно передать словарь весов токенов для учета похожести слов внутри одной категории
   - в этом случае будет учитыватся степень сходства каждого слова (0,1) умноженное на коэффициент из словаря
   - коэффициенты из словаря должны быть от 0 до 1
   - по умолчанию коэффициенты для отсутсвующих тегов равны 0
  '''

  if len(doc['ents'])+len(doc1['ents']) == 0:
     return 1

  c=0
  words=[]
  for i in doc['ents']:
    for j in doc1['ents']: 
      if i['word']==j['word']:
        words.append(i['word'])
        c+=1


  l1=[i for i in doc['ents'] if i['word'] not in words]
  l2=[i for i in doc1['ents'] if i['word'] not in words]

  numerator=0
  denumerator=0

  for i in l1:
    max_num=0
    for j in l2:
      if i['label']==j['label']:
        cos = dl.SequenceMatcher(lambda x: x == "",i['word'],j['word']).ratio()
        max_num = cos if cos>max_num else max_num
    numerator+= max_num * (0 if not (i['label'] in label_impotance.keys()) else label_impotance[i['label']])


    denumerator+=1

  for i in l2:
    max_num=0
    for j in l1:
      if i['label']==j['label']:
        cos = dl.SequenceMatcher(lambda x: x == "",i['word'],j['word']).ratio()
        max_num = cos if cos>max_num else max_num
    numerator+=max_num * (0 if not (i['label'] in label_impotance.keys()) else label_impotance[i['label']])
    denumerator+=1

  return 0 if denumerator+c==0 else (c+numerator)/(denumerator+c)


def get_ner_tokens(text, nlp, conf_for_bert=0.8, label_drop_list=[]):
  '''
  Возращает словарь из текста предложения и токенов в нем
  На вход кушает текст и nlp из huggingface
  '''

  def rubert_to_lst_dict(rubert):
    r=list(rubert.copy())
    for i in r:
      i['label'] = i['entity_group']
      i['score'] =float(i['score'])
    return r


  def cut_conf_less_than(dct,conf):
    return [i for i in dct if i['score']>conf]


  def cut_tag(dct,lst):
    return [i for i in dct if (i['label'] not in lst)]


  ans=cut_tag(cut_conf_less_than(rubert_to_lst_dict(nlp(text)),conf_for_bert),label_drop_list) 
  ans={
      'text' : text,
      'ents' : ans
      }

  return ans


def calculate_importance(sim_score):
    if sim_score < 0.65:
        return 5
    elif sim_score < 0.70:
        return 4
    elif sim_score < 0.82:
        return 3
    elif sim_score <= 0.94:
        return 2
    elif sim_score <= 1:
        return 1
    return 1


def calculate_importance_alone(score):
    if score < 0.1:
        return 1
    elif score < 0.2:
        return 2
    elif score < 0.35:
        return 3
    elif score <= 0.42:
        return 4
    elif score <= 2:
        return 5
    return 1


def entity_extract(sentanse: str) ->dict:
    '''
    выделение сущностей из предложения
    '''
    # morph_vocab = MorphVocab()
    # names_extractor = NamesExtractor(morph_vocab)
    # dates_extractor = DatesExtractor(morph_vocab)
    # money_extractor = MoneyExtractor(morph_vocab)
    # addr_extractor = AddrExtractor(morph_vocab)


    # extractors=[dates_extractor,money_extractor,addr_extractor]

    # doc=get_ner_tokens(sentanse,tokenizer,model_ent,nlp,extractors,conf_for_bert=0.7,label_drop_list=['LOC'])
    doc=get_ner_tokens(sentanse,nlp,conf_for_bert=0.7,label_drop_list=[])

    return doc


def count_part_of_speech(sen):
    '''
    принимает на вход предложение (строку)
    вычисляет концентрацию ADJ, VERB
    '''
    imp_part_of_speech = ['ADJ','VERB', 'PROPN']
    document = nlp_spacy(sen)
    count = Counter([el.pos_ for el in document])
    return sum([count.get(el,0) for el in imp_part_of_speech])/len(document.text.split())


def get_tag_score(doc,label_impotance={}):
  '''
  Принимает словарь из get_ner_tokens()
  Возращает оценку, основанную на количестве и важности токенов
  Можно передать словарь весов токенов для каждого токена
  '''
  a=0

  for i in label_impotance_alone.keys():
    a+=label_impotance[i]
  x=0
  for i in doc['ents']:
    x+= (0 if not (i['label'] in label_impotance.keys()) else label_impotance[i['label']])

  return math.exp(-x/a)


def get_json(t1,t2,d_eq,d_changed,deleted):
    '''
    формирование файла разметки.

    '''
    out = []
    d_finaly = {}

    for k,v in t2.items():
        d_finaly = {}
        d_finaly["id"] = k
        d_finaly["text"] = v[0]
        d_finaly["num_paragraph"] = v[1]

        if d_eq.get(k) is not None:
            d_finaly["score"] = 1
            d_finaly["n_matches"] = d_eq[k]
            d_finaly["importance"] = 0
        elif d_changed.get(k) is not None:
            tags_1 = entity_extract(t1[d_changed[k]][0])
            tags_2 = entity_extract(v[0])

            sim_score = round(get_sent_similarity(v[0],t1[d_changed[k]][0]).item(),3)
            d_finaly["sim_score"] = sim_score
            d_finaly["entity_score"] = get_tag_diff_score(tags_1,tags_2,label_impotance={})
            d_finaly["check_test"] = t1[d_changed[k]]
            d_finaly["check_test_entities"] = tags_1['ents']
            d_finaly["n_matches"] = d_changed[k]
            d_finaly['entities'] = tags_2['ents']
            d_finaly["markdown"] = get_diff_to_html(t1[d_changed[k]][0], v[0]) # разметка для двух текстов
            d_finaly["importance"] = calculate_importance(sim_score) # 1-5
            d_finaly["markdown_ent_1"] = displacy.render(tags_1, style="ent",manual=True) # разметка выделения сущностей 1 текст необходимо удалить \
            d_finaly["markdown_ent_2"] = displacy.render(tags_2, style="ent",manual=True) # разметка выделения сущностей 2-й текст

        else: # добавленные предложения
            score_alone = 1 - get_tag_score(entity_extract(v[0]),label_impotance_alone) + count_part_of_speech(v[0])
            d_finaly["score"] = 0
            d_finaly["n_matches"] = False
            d_finaly["entities"] = entity_extract(v[0])['ents']
            d_finaly["importance"] = calculate_importance_alone(score_alone)
            d_finaly["markdown_ent"] = displacy.render(entity_extract(v[0]), style="ent",manual=True)

        out.append(d_finaly)

    d_finaly = {}
    d_finaly['eq_and_match'] = out

    del_lst = []
    for d in deleted:
        del_out = {}
        score_alone = 1 - get_tag_score(entity_extract(t1[d][0]),label_impotance_alone) + count_part_of_speech(t1[d][0])
        ents = entity_extract(t1[d][0])
        del_out["id"] = d
        del_out["text"] = t1[d][0]
        del_out["num_paragraph"] = t1[d][1]
        del_out['score'] = 0
        del_out["n_matches"] = False
        del_out["entities"] = ents['ents']
        del_out["importance"] = calculate_importance_alone(score_alone)
        del_out["markdown_ent"] = displacy.render(ents, style="ent",manual=True)
        del_lst.append(del_out)

    d_finaly['deleted'] = del_lst

    len_t2 = 0
    len_t1 = 0
    num_eq = 0
    num_added =0
    matched_sim_score = []
    matched_ent_score = []
    added_importance = []
    del_importance = []
    matched_importance = []

    for el in d_finaly['eq_and_match']:
        num_eq+=el.get('score',0)
        if el.get('sim_score') is not None:
            matched_sim_score.append(el.get('sim_score'))
            matched_ent_score.append(el.get('entity_score'))
            matched_importance.append(el.get('importance'))
        elif el.get('score') is not None and el.get('score')==0:
            num_added+=1
            added_importance.append(el.get('importance'))

    for el in d_finaly['deleted']:
        del_importance.append(el.get('importance'))



    analytics = {
        'num_sentence_t1': len(t1), # предложений в 1 тексте
        'num_sentence_t2': num_eq+len(matched_sim_score)+num_added, # предложений в 2 тексте
        'num_equal': num_eq, # кол-во одинаковых предложений
        'num_added': num_added, # кол-во добавленных предложений
        'num_deleted': len(del_importance), # кол-во удаленных предложений
        'num_changed': len(matched_importance),
        'mean_sim_score': sum(matched_sim_score)/len(matched_sim_score), # средний симиларити скор
        'median_sim_score': statistics.median(matched_sim_score), # медиана симиларити скор
        'mean_ent_score': sum(matched_ent_score)/len(matched_ent_score), # средний скор похожести сущностей
        'median_ent_score': statistics.median(matched_ent_score),# медианный скор похожести сущностей

        'mean_matched_importance': sum(matched_importance)/len(matched_importance), # средний важность изменений в соответствующих предложениях
        'median_ent_score': statistics.median(matched_importance), # медианный важность изменений в соответствующих предложениях

        'mean_del_importance': sum(del_importance)/len(del_importance), # средняя важность удаленных предложений
        'median_del_importance': statistics.median(del_importance), # медианная важность удаленных предложений

        'mean_added_importance': sum(added_importance)/len(added_importance), # средняя важность у удаленных предложений
        'median_added_importance': statistics.median(added_importance), # медианная важность у удаленных предложений


        }

    # with open('markdown.json', 'w') as outfile:
    #     json.dump(d_finaly, outfile, ensure_ascii=False)

    # with open('text1.json', 'w') as outfile:
    #     json.dump(t1, outfile, ensure_ascii=False)

    # with open('analytics.json', 'w') as outfile:
    #     json.dump(analytics, outfile, ensure_ascii=False)

    return d_finaly, t1, analytics # раньше возвращалось только 2 аргумента


if __name__ =='__main__':
    t1 = get_all_text("11.docx") # текст 1 путь к файлу (doc,docx,rtf) STRING
    t2 = get_all_text("22.docx") # текст 2 путь к файлу (doc,docx,rtf) STRINGv
    d_eq, d_changed = get_match(t1,t2) # словарь полных совпадений и изменений {text1_id: text2_id}
    deleted, added = get_minus_and_plus(t1,t2,d_eq,d_changed) # удаленные из 1 текста и добавленные во 2 текст
    get_json(t1,t2,d_eq,d_changed,deleted) # формирование файла разметки
