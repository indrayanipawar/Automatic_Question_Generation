from flask import Flask, render_template, request, redirect
import json  
from flask_cors import CORS

import spacy
from spacy import displacy
import pandas as pd
import random
from pathlib import Path
import sklearn
app = Flask(__name__,template_folder='../templates')
nlp = spacy.load('en_core_web_sm')

from flask import Flask, render_template, request, redirect
import json  
from flask_cors import CORS

from summarizer import Summarizer
import pprint
import itertools
import re
import pke
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import nltk
# nltk.download('stopwords')
# nltk.download('popular')

import requests
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk
from nltk.corpus import wordnet as wn
import pickle

def startgeneration(text, option):
    #print ('\nChoose from the following : \n1. Fill in the Blanks \n2. Multiple Choice Questions \n3. True or False Questions')
    if option == 'blanks' or option == 'truefalse':
        #print('Generating ' + option)
        return generateQuestions(text, option)

    elif option == 'mcq':
        model = Summarizer()
        result = model(text, min_length=60, max_length = 500 , ratio = 0.4)
        summarized_text = ''.join(result)
        keywords = get_nouns_multipartite(text)
        filtered_keys=[]
        for keyword in keywords:
            if keyword.lower() in summarized_text.lower():
                filtered_keys.append(keyword)

        sentences = tokenize_sentences(summarized_text)
        keyword_sentence_mapping = get_sentences_for_keyword(filtered_keys, sentences)
        key_distractor_list = {}
        for keyword in keyword_sentence_mapping:
            wordsense = get_wordsense(keyword_sentence_mapping[keyword][0],keyword)
            if wordsense:
                distractors = get_distractors_wordnet(wordsense,keyword)
                if len(distractors) ==0:
                    distractors = get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    key_distractor_list[keyword] = distractors
            else:
                distractors = get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    key_distractor_list[keyword] = distractors

        index = 1
        MCQ = []
        for each in key_distractor_list:
            Que = {}
            sentence = keyword_sentence_mapping[each][0]
            pattern = re.compile(each, re.IGNORECASE)
            output = pattern.sub( " _______ ", sentence)
            #print ("%s)"%(index),output)
            Que['Question'] = str(index) + ") " + output
            choices = [each.capitalize()] + key_distractor_list[each]
            top4choices = choices[:4]
            random.shuffle(top4choices)
            optionchoices = ['a','b','c','d']
            Que['Answer'] = []
            
            for idx,choice in enumerate(top4choices):
                Que['Answer'].append({optionchoices[idx] : choice})
                #print ("\t",optionchoices[idx],")"," ",choice)
                Que['MoreOptions'] = choices[4:20]
            #print ("\nMore options: ", choices[4:10],"\n\n")
            MCQ.append(Que)
            index = index + 1
        return MCQ
        


def dumpPickle(fileName, content):
    pickleFile = open(fileName, 'wb')
    pickle.dump(content, pickleFile, -1)
    pickleFile.close()

def loadPickle(fileName):    
    file = open(fileName, 'rb')
    content = pickle.load(file)
    file.close()
    return content

def pickleExists(fileName):
    file = Path(fileName)
    if file.is_file():
        return True
    return False

def extractAnswers(qas, doc):
    answers = []

    senStart = 0
    senId = 0

    for sentence in doc.sents:
        senLen = len(sentence.text)

        for answer in qas:
            answerStart = answer['answers'][0]['answer_start']

            if (answerStart >= senStart and answerStart < (senStart + senLen)):
                answers.append({'sentenceId': senId, 'text': answer['answers'][0]['text']})

        senStart += senLen
        senId += 1
    return answers

def tokenIsAnswer(token, sentenceId, answers):
    for i in range(len(answers)):
        if (answers[i]['sentenceId'] == sentenceId):
            if (answers[i]['text'] == token):
                return True
    return False

def getNEStartIndexs(doc):
    neStarts = {}
    for ne in doc.ents:
        neStarts[ne.start] = ne
    return neStarts 

def getSentenceStartIndexes(doc):
    senStarts = []
    for sentence in doc.sents:
        senStarts.append(sentence[0].i)
    return senStarts

def getSentenceForWordPosition(wordPos, senStarts):
    for i in range(1, len(senStarts)):
        if (wordPos < senStarts[i]):
            return i - 1


def addWordsForParagrapgh(newWords, text):
    doc = nlp(text)
    neStarts = getNEStartIndexs(doc)
    senStarts = getSentenceStartIndexes(doc)

    #index of word in spacy doc text
    i = 0

    while (i < len(doc)):
        #If the token is a start of a Named Entity, add it and push to index to end of the NE
        if (i in neStarts):
            word = neStarts[i]
            #add word
            currentSentence = getSentenceForWordPosition(word.start, senStarts)
            wordLen = word.end - word.start
            shape = ''
            for wordIndex in range(word.start, word.end):
                shape += (' ' + doc[wordIndex].shape_)

            newWords.append([word.text,
                            0,
                            0,
                            currentSentence,
                            wordLen,
                            word.label_,
                            None,
                            None,
                            None,
                            shape])
            i = neStarts[i].end - 1
        #If not a NE, add the word if it's not a stopword or a non-alpha (not regular letters)
        else:
            if (doc[i].is_stop == False and doc[i].is_alpha == True):
                word = doc[i]

                currentSentence = getSentenceForWordPosition(i, senStarts)
                wordLen = 1

                newWords.append([word.text,
                                0,
                                0,
                                currentSentence,
                                wordLen,
                                None,
                                word.pos_,
                                word.tag_,
                                word.dep_,
                                word.shape_])
        i += 1

def oneHotEncodeColumns(df):
    columnsToEncode = ['NER', 'POS', "TAG", 'DEP']

    for column in columnsToEncode:
        one_hot = pd.get_dummies(df[column])
        one_hot = one_hot.add_prefix(column + '_')

        df = df.drop(column, axis = 1)
        df = df.join(one_hot)
    return df

def generateDf(text):
    words = []
    addWordsForParagrapgh(words, text)

    wordColums = ['text', 'titleId', 'paragrapghId', 'sentenceId','wordCount', 'NER', 'POS', 'TAG', 'DEP','shape']
    df = pd.DataFrame(words, columns=wordColums)

    return df

def prepareDf(df):
    #One-hot encoding
    wordsDf = oneHotEncodeColumns(df)

    #Drop unused columns
    columnsToDrop = ['text', 'titleId', 'paragrapghId', 'sentenceId', 'shape']
    wordsDf = wordsDf.drop(columnsToDrop, axis = 1)

    #Add missing colums 
    predictorColumns = ['wordCount', 'NER_CARDINAL', 'NER_DATE', 'NER_EVENT', 'NER_FAC',
       'NER_GPE', 'NER_LANGUAGE', 'NER_LAW', 'NER_LOC', 'NER_MONEY',
       'NER_NORP', 'NER_ORDINAL', 'NER_ORG', 'NER_PERCENT', 'NER_PERSON',
       'NER_PRODUCT', 'NER_QUANTITY', 'NER_TIME', 'NER_WORK_OF_ART', 'POS_ADJ',
       'POS_ADP', 'POS_ADV', 'POS_CCONJ', 'POS_DET', 'POS_NOUN', 'POS_NUM',
       'POS_PROPN', 'POS_SCONJ', 'POS_VERB', 'TAG_CC', 'TAG_CD', 'TAG_IN',
       'TAG_JJ', 'TAG_JJR', 'TAG_JJS', 'TAG_NN', 'TAG_NNP', 'TAG_NNPS',
       'TAG_NNS', 'TAG_PDT', 'TAG_RB', 'TAG_RBR', 'TAG_RBS', 'TAG_VB',
       'TAG_VBD', 'TAG_VBG', 'TAG_VBN', 'TAG_VBP', 'TAG_VBZ', 'DEP_ROOT',
       'DEP_acl', 'DEP_acomp', 'DEP_advcl', 'DEP_advmod', 'DEP_agent',
       'DEP_amod', 'DEP_appos', 'DEP_attr', 'DEP_aux', 'DEP_auxpass',
       'DEP_ccomp', 'DEP_compound', 'DEP_conj', 'DEP_csubj', 'DEP_csubjpass',
       'DEP_dative', 'DEP_dep', 'DEP_dobj', 'DEP_nmod', 'DEP_npadvmod',
       'DEP_nsubj', 'DEP_nsubjpass', 'DEP_nummod', 'DEP_oprd', 'DEP_pcomp',
       'DEP_pobj', 'DEP_poss', 'DEP_predet', 'DEP_prep', 'DEP_relcl',
       'DEP_xcomp']
    for feature in predictorColumns:
        if feature not in wordsDf.columns:
            wordsDf[feature] = 0
    return wordsDf

def predictWords(wordsDf, df):
    predictorPickleName = 'C:/Users/Indrayani/Downloads/GenerateQue/GenerateQue/nb-predictor.pkl'
    predictor = loadPickle(predictorPickleName)
    y_pred = predictor.predict_proba(wordsDf)

    labeledAnswers = []
    for i in range(len(y_pred)):
        labeledAnswers.append({'word': df.iloc[i]['text'], 'prob': y_pred[i][0]})
    return labeledAnswers

def blankAnswer(firstTokenIndex, lastTokenIndex, sentStart, sentEnd, doc):
    leftPartStart = doc[sentStart].idx
    leftPartEnd = doc[firstTokenIndex].idx
    rightPartStart = doc[lastTokenIndex].idx + len(doc[lastTokenIndex])
    rightPartEnd = doc[sentEnd - 1].idx + len(doc[sentEnd - 1])
    question = doc.text[leftPartStart:leftPartEnd] + '_____' + doc.text[rightPartStart:rightPartEnd]

    return question

def addQuestions(answers, text):
    doc = nlp(text)
    currAnswerIndex = 0
    qaPair = []

    #Check wheter each token is the next answer
    for sent in doc.sents:
        for token in sent:

            #If all the answers have been found, stop looking
            if currAnswerIndex >= len(answers):
                break

            #In the case where the answer is consisted of more than one token, check the following tokens as well.
            answerDoc = nlp(answers[currAnswerIndex]['word'])
            answerIsFound = True

            for j in range(len(answerDoc)):
                if token.i + j >= len(doc) or doc[token.i + j].text != answerDoc[j].text:
                    answerIsFound = False
            #If the current token is corresponding with the answer, add it 
            if answerIsFound:
                question = blankAnswer(token.i, token.i + len(answerDoc) - 1, sent.start, sent.end, doc)

                qaPair.append({'question' : question, 'answer': answers[currAnswerIndex]['word'], 'prob': answers[currAnswerIndex]['prob']})
                currAnswerIndex += 1
    return qaPair

def sortAnswers(qaPairs):
    orderedQaPairs = sorted(qaPairs, key=lambda qaPair: qaPair['prob'])
    return orderedQaPairs

def generate_distractors(answer, count):
    answer = str.lower(answer)
    ##Extracting closest words for the answer. 
    try:
        closestWords = model.most_similar(positive=[answer], topn=count)
    except:
        return []
    distractors = list(map(lambda x: x[0], closestWords))[0:count]
    return distractors

def addDistractors(qaPairs, count):
    for qaPair in qaPairs:
        distractors = generate_distractors(qaPair['answer'], count)
        qaPair['distractors'] = distractors
    return qaPairs


def generateQuestions(text, option):
    # Extract words
    df = generateDf(text)
    wordsDf = prepareDf(df)

    # Predict
    labeledAnswers = predictWords(wordsDf, df)

    # Transform questions
    qaPairs = addQuestions(labeledAnswers, text)

    # Pick the best questions
    orderedQaPairs = sortAnswers(qaPairs)

    # Generate distractors
    questions = addDistractors(orderedQaPairs[:len(orderedQaPairs)], 3)

    # Formating the output into a txt file
    path = 'generatedQuestions.txt'
    output_data = list()
    #file = open(path, 'w')
    if (option == 'blanks'):
        for i in range(len(orderedQaPairs)): 
            output_data.append({
                'Question': questions[i]['question'],
                'Answer': questions[i]['answer']
            })

    if (option == 'truefalse'):
        for i in range(len(orderedQaPairs)):
            options = []
            options.append(questions[i]['answer'])

            for distractor in questions[i]['distractors']:
                options.append(distractor)
            random.shuffle(options)
            inputOption = random.choice(options) 
            if (inputOption == questions[i]['answer']):
                Ans = 'True'
            else:
                Ans = 'False'
            inputOption = random.choice(options)
            output_data.append({
                'Question': questions[i]['question'].replace('_____', inputOption),
                'Answer': Ans
            })
    return output_data

path = 'C:/Users/Indrayani/Downloads/GenerateQue/GenerateQue/distractor.pkl'
model = loadPickle(path)





############################# MCQ Code #################################

def get_nouns_multipartite(text):
    out=[]

    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input= text)
    #    not contain punctuation marks or stopwords as candidates.
    pos = {'PROPN'}
    #pos = {'VERB', 'ADJ', 'NOUN'}
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    # 4. build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.
    extractor.candidate_weighting(alpha=1.1,
                                  threshold=0.75,
                                  method='average')
    keyphrases = extractor.get_n_best(n=20)

    for key in keyphrases:
        out.append(key[0])

    return out

def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences


# Distractors from Wordnet
def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

def get_wordsense(sent,word):
    word= word.lower()
    if len(word.split())>0:
        word = word.replace(" ","_")

    synsets = wn.synsets(word,'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output =  adapted_lesk(sent, word, pos='n')
        lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

# Distractors from http://conceptnet.io/
def get_distractors_conceptnet(word):
    word = word.lower()
    original_word= word
    if (len(word.split())>0):
        word = word.replace(" ","_")
    distractor_list = [] 
    url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
    obj = requests.get(url).json()

    for edge in obj['edges']:
        link = edge['end']['term'] 

        url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)
    return distractor_list



app = Flask(__name__,template_folder='../templates')
CORS(app)
output = {"data": "This is output"}
@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        text = request.form['input']
        type_que = request.form['Qtype']
        #print(startgeneration(text,type_que))
        return render_template('home.html', option = type_que, data = startgeneration(text,type_que))
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
