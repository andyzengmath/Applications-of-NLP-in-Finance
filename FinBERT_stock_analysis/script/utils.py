# FinBert

import torch.nn.functional as F
def SentimentAnalyzer(doc):
    pt_batch = tokenizer(doc,padding=True,truncation=True,max_length=512,return_tensors="pt")
    outputs = model(**pt_batch)
    pt_predictions = F.softmax(outputs.logits, dim=-1)
    return pt_predictions.detach().cpu().numpy()

ONE_DAY = datetime.timedelta(days=1)
HOLIDAYS_US = holidays.US()
def next_business_day(dateString):
    datetimeObj = datetime.datetime.strptime(dateString, '%Y-%m-%d')
    next_day = datetimeObj + ONE_DAY
    while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_US:
        next_day += ONE_DAY
    return next_day

def findPercentageBySentences(sentenceList):
    posAvg, negAvg, neuAvg = 0, 0, 0
    sentimentArr = SentimentAnalyzer(sentenceList)
    sentimentArr = np.mean(sentimentArr, axis=0)
    posAvg=sentimentArr[0]
    negAvg=sentimentArr[1]
    neuAvg=sentimentArr[2]
    return {'numArticles': len(sentenceList), 'pos': posAvg, 'neg': negAvg, 'neu' : neuAvg}

# LTSM

def lemmatize(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')


def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower()

    return [lemmatize(token) for token in gensim.utils.simple_preprocess(words) ]

def SentimentAnalyzer(doc):
    embeddings = np.array([])
    for i in doc:
        i=' '.join(preprocess(i))
        embedding=[]
        with open('../input/lstmmodel/dict.pkl', 'rb') as handle:
            corpus_tfidf_vectorizer=joblib.load(handle)
        corpus_vocabulary=corpus_tfidf_vectorizer.vocabulary_
        sent_emb=np.zeros(487)
        intersection = np.intersect1d(i.split(), list(corpus_vocabulary.keys()))
        sent_emb = list(map(corpus_vocabulary.get, intersection.tolist()))
        embedding.append(sent_emb)
        embeddings=np.append(embeddings, np.array(embedding))
    return model.predict(embeddings, batch_size=128)