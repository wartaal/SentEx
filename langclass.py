from collections import Counter
import pickle
import re
from sklearn.feature_extraction import DictVectorizer
#from sklearn import linear_model
from sklearn import svm


class langguesser: 
    
    def ngram(self,string,n):
        liste = []
        if n < len(string):
            for p in range(len(string) - n + 1) :
                tg = string[p:p+n]
                liste.append(tg)
        return liste
    
    def xgram(self,string):
        return [w for n in range(1,5) for w in self.ngram(string,n)]
    
    
    def features(self,text):
        text = re.sub(r'[^\w\- ,\.;!?]+', '', text)
        text = re.sub(r' . ', ' ', text)
        text = re.sub(r' +', ' ', text)
        xg = self.xgram(text)
        model = Counter(xg)  
        nr_of_ngs = len(xg)

        for w in model:
            model[w] = float(model[w]) / float(nr_of_ngs)

        return xg, model
       
    def train(self):
        featcount = Counter()
        labels = []
        features = []
        
        file_de = open('train/WikiSentences_de.txt','r',encoding = 'utf8')
        for line in file_de:
            txt = line.strip()
            labels.append('de')
            xg, feats = self.features(txt)
            features.append(feats)
            featcount.update(xg)
        file_de.close()


        file_en = open('train/WikiSentences_en.txt','r',encoding = 'utf8')
        for line in file_en:
            txt = line.strip()
            labels.append('en')
            xg, feats = self.features(txt)
            features.append(feats)
            featcount.update(xg)
        file_en.close()
        
        #Select most common xgrams 
        n = sum([v for _,v in featcount.most_common()])
        ln = len(featcount)
        for lim in range(50,ln):
            n1 =  sum(v for _,v in featcount.most_common(lim))
            if n1/n > 0.96:
                break
        usefull = set([f for f,_ in featcount.most_common(lim)])
        mf = featcount.most_common(lim)[-1][1]
        print('reducing from ' + str(ln) + ' to ' + str(lim) + ' features with min. freq. of ' + str(mf))
        
        #Reduce features
        print('Building feature vectors.')
        features = [{f:v for (f,v) in ex.items() if f in usefull} for ex in features]
   
        self.d2v = DictVectorizer(sparse=True)
        feat_vec = self.d2v.fit_transform(features)
    
        print('Training classifier.')
        #self.classif = linear_model.LogisticRegression(class_weight = 'balanced')
        self.classif = svm.SVC(kernel = 'linear',class_weight = 'balanced')
        self.classif.fit(feat_vec,labels)
    
        pickle.dump((self.d2v, self.classif), open('langclassmodels.p', 'wb'))

   
  
    def load(self,f):
        (self.d2v, self.classif) = pickle.load(open(f, 'rb'))



   
    def identify(self,text):
        _,features = self.features(text)
        featvec = self.d2v.transform(features)
        lang = self.classif.predict(featvec)[0]
        return lang

