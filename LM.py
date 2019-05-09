import nltk
from nltk.lm import MLE, Laplace, KneserNeyInterpolated
from nltk.corpus import reuters
nltk.download('reuters')
from collections import Counter
from nltk import bigrams, trigrams



text = reuters.sents()


text=[[j.lower() for j in i] for i in text]
text = [[''.join(c for c in s if c not in ["."]) for s in k]for k in text]

from nltk.lm.preprocessing import pad_both_ends

y=(map((lambda x:pad_both_ends(x,n=2)),text))
y=list(map(lambda x: list(x),y))


from nltk.lm.preprocessing import flatten
from nltk.util import everygrams
bigramsList=list(map(lambda x: list(trigrams(x)),y))
bigramsList=list(flatten(bigramsList))
#list(everygrams(bigramsList, max_len=2))


vocab=list(flatten(pad_both_ends(sent, n=2) for sent in text))
from nltk.lm import Vocabulary
vocab =list(Vocabulary(vocab, unk_cutoff=1))

'''
from nltk.lm.preprocessing import padded_everygram_pipeline
train, vocab = padded_everygram_pipeline(2, text)
'''


lm = Laplace(3)
lm.fit([bigramsList], vocabulary_text=list(vocab))




lm.generate(4,text_seed=["government","had"])



def generateSentences(v):
    sent=v
    v=[lm.generate(1,text_seed=v)]
    sent=sent+v
    while v[0]!='</s>':
        l=len(sent)
        v=[lm.generate(1,text_seed=[sent[l-2],sent[l-1]])]
        sent=sent+v
    return sent



    
    

sen=generateSentences(['<s>','india'])
sen=" ".join(sen)
print (sen)

x=[]
lm.entropy(bigramsList)
