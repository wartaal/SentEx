import codecs
import xml.sax
import nltk
import re
from collections import Counter
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import glob
import os
import pickle
import json
import langclass

classif = langclass.langguesser()
classif.load('langclassmodels.p')


with open('wfreq.json', 'r') as f:
    commonwords = json.load(f)


class BookHandler(xml.sax.ContentHandler):
    
    def __init__(self):
        self.in_txt = False #true wenn zwischen <txt> und </txt>
        self.in_sup = False
        self.text = ''
        self.fontsizes = Counter()
        self.total_len = 0
        self.attributes = {}
        self.fntext = ''
        self.footnotes = {}
        self.footnotesA = []
        self.pagenr = 0
    
    
    def startElement(self,name,attrs):
        if name == "txt":
            self.in_txt = True
            self.text = ""
            self.attributes = attrs
        elif name == "sup":
            self.in_sup = True
        elif name == "page":
            self.pagenr = attrs['nr']
            self.footnotesA = []
       
    def characters(self,c):
        if self.in_txt:
            self.text += c
        if self.in_sup:
            self.fntext += c
    
    def endElement(self, name):
        if name == "txt":
            self.in_txt = False
            l = len(self.text)
            self.total_len += l
            if self.attributes['italic'] == '0' and self.attributes['bold'] == '0':
                size = self.attributes['fontsize']
                self.fontsizes.update({size:l})          
            for fn in self.footnotesA:
                for line in self.text.split('\n'):
                    if line.strip().startswith(fn):
                        fns = self.footnotes.get(self.pagenr,[])
                        fns.append(fn)
                        self.footnotes[self.pagenr] = fns
                        break
        
        elif name == "sup":
            self.in_sup = False
            if self.fntext not in self.footnotesA:
                self.footnotesA.append(self.fntext)
            self.fntext = ""
        
    def get_common_fsizes(self):
        result = []
        percentage = Counter({size:freq/self.total_len for size,freq in self.fontsizes.items()})
        #print(percentage.most_common(5))
        sigma = 0
        for s,p in percentage.most_common():
            result.append(s)
            sigma += p
            if sigma > 0.55 or len(result) > 4:
                break
        return result

class TextExtract(xml.sax.ContentHandler):
    
    def __init__(self):
        self.in_txt = False #true wenn zwischen <txt> und </txt>
        self.in_sup = False
        self.text = ''
        self.booktext = ''
        self.last_fs = ''
        self.attributes = {}
        self.texttypes = {}
        self.len_extracted = 0
        self.charmap = {'\(cid:5\)':'-',
              '\(cid:4\)':'≈',
              '\(cid:11\)':'Ψ'}
        self.footnotes = {}
        self.pagenr = 0
        self.fntext = ''
        self.sentences = []
    
    def replaceCID(self,txt):
        for cid in self.charmap:
            txt = re.sub(cid,self.charmap[cid],txt)
        return txt
    
    def replaceLigature(self,txt):
        txt =  re.sub('ﬁ','fi',txt)
        txt =  re.sub('ﬂ','fl',txt)
        txt =  re.sub('ﬀ','ff',txt)
        txt =  re.sub('­','-',txt)
    
        return txt
    
    def repairUmlaut(self,txt):
        txt =  re.sub('¨a','ä',txt)
        txt =  re.sub('¨u','ü',txt)
        txt =  re.sub('¨o','ö',txt)
        txt =  re.sub('¨A','Ä',txt)
        txt =  re.sub('¨U','Ü',txt)
        txt =  re.sub('¨O','Ö',txt)
        txt =  re.sub('˜a','ä',txt)
        txt =  re.sub('˜u','ü',txt)
        txt =  re.sub('˜o','ö',txt)
        txt =  re.sub('˜A','Ä',txt)
        txt =  re.sub('˜U','Ü',txt)
        txt =  re.sub('˜O','Ö',txt)


        return txt
  
    def replaceSpaces(self,txt): 
    #Specific for German abbreviations
        txt = re.sub(r'\s+',' ',txt)
        
        txt = re.sub(r'z\. B\.','z.B.',txt)
        txt = re.sub(r'Z\. B\.','Z.B.',txt)
        txt = re.sub(r'u\. a\.','u.a.',txt)
        txt = re.sub(r'U\. a\.','U.a.',txt)
        txt = re.sub(r'd\. h\.','d.h.',txt)
        txt = re.sub(r'D\. h\.','D.H.',txt)
        txt = re.sub(r'z\. T\.','z.T.',txt)
        txt = re.sub(r'Z\. T\.','Z.T.',txt)
        txt = re.sub(r'o\. g\.','o.g.',txt)
        txt = re.sub(r'O\. g\.','O.g.',txt)
        txt = re.sub(r'u\. U\.','u.U.',txt)
        txt = re.sub(r'U\. U\.','U.U.',txt)       
        
        return txt
    
    def setTextType(self,types):
        for t in types:
            self.texttypes[t] = ""
            
    def setFootnotes(self,fns):
        self.footnotes = fns
            
    def remove_line_breaks(self,text):
    #Specific for German as well, to handle coordinated compounds  with hyphens and "und" or  "oder".
        result = ""
        for line in text.split('\n'):
            line = line.strip(' \n\t\r')
            if len(result) < 2:
                result += line
            elif result[-1] == '-':
                if len(result) > 10 and len(line) > 2 and line[0] in 'abcdefghijklmnopqrstuvwxyzäüöß' and line[1] in 'abcdefghijklmnopqrstuvwxyzäüö' and not line.startswith('und') and not line.startswith('oder'):
                      result = result[:-1] + line
                elif len(line) > 0:
                     result += line
            else:
                if len(line) > 0:
                    space = result.rfind(' ')
                    lastw = result[space+1:].lower()
                    space = line.find(' ')
                    firstw = line[:space]
                    if len(firstw) > 2 and firstw[-1] in ';:,.?!':
                        firstw = firstw[:-1]
                    if len(firstw) < 2 or len(lastw) < 2:
                        result += ' '
                    elif commonwords.get(lastw,0) > 50 and commonwords.get(firstw,0) > 50:
                        result += ' '
                    elif not re.fullmatch('[a-zäüöß]+',firstw) or not re.fullmatch('[a-zäüöß]+',lastw):
                        result += ' '
                    elif not lastw+firstw in commonwords:
                        result += ' '
                    else:
                        print(lastw,firstw)
                result += line
        return result
    
    def clean(self,text):
        if len(text.strip()) < 10:
            return False
        
        nrOfLetters = 0
        for c in text:
            if c.lower() in 'abcdefghojklmnopqrstuvwxyzäüöß':
                nrOfLetters+=1
        if nrOfLetters/len(text) < 0.5:
            return False
        
        return True
    
    def clean_sent(self,s):
        if len(s.strip()) < 36:
            return False
        
        nrOfLetters = 0
        for c in s:
            if c.lower() in 'abcdefghojklmnopqrstuvwxyzäüöß':
                nrOfLetters+=1
        if nrOfLetters/len(s) < 0.5:
            return False
        
        if "cid:" in s:
            return False
        
        if s[0] not in "ABCDEFGHIJKLMNOPQRSTUVWZYZÖÄÜ":
            return False
        
        if s[-1] not in ".:?!":
            return False
        
        if s.count('(') != s.count(')'):
            return False
        
        return True
        
    
    def extract(self,rawtext):
        txt = self.replaceCID(rawtext)
        txt = self.replaceLigature(txt)
        txt = self.repairUmlaut(txt)
        txt = self.remove_line_breaks(txt)
        txt = self.replaceSpaces(txt) #after remove linebreaks! 
        if self.clean(txt):
            if len(self.booktext) > 0:
                self.booktext += '\n'
            self.booktext += txt
    
    def startElement(self,name,attrs):
        if name == "txt":
            self.in_txt = True
            self.text = ""
            self.attributes = attrs
        elif name == "page":
            self.pagenr = attrs['nr']
        elif name == "sup":
            self.in_sup = True
       
    def characters(self,c):
        if self.in_sup:
            self.fntext += c
        elif self.in_txt:
            self.text += c
        
    
    def endElement(self, name):
        if name == "txt":
            self.in_txt = False
            if self.attributes['italic'] == '0' and self.attributes['bold'] == '0' and self.attributes['fontsize'] in self.texttypes:
                fs = self.attributes['fontsize']
                if self.last_fs == fs or (self.text[-1].isupper() and len(self.texttypes[fs]) > 1 and self.texttypes[fs][-1] in '.:;!?'):
                    self.extract(self.texttypes[fs])
                    #print(fs," -- ",len(self.texttypes[fs]))
                    self.texttypes[fs] = self.text
                else:
                    self.texttypes[fs] += self.text
                
            self.last_fs = self.attributes['fontsize']
        elif name == "sup":
            self.in_sup = False
            if not self.fntext in self.footnotes.get(self.pagenr,[]):
                self.text += self.fntext 
            self.fntext = ''
        elif name == "extracted":
            for fs in self.texttypes:
                self.extract(self.texttypes[fs])
            trainer = PunktTrainer()
            trainer.train(self.booktext, finalize=False, verbose=False)
            abbreviations =  "Abschn., Kap., Abb., Nr., s., S."
            trainer.train(abbreviations, finalize=True, verbose=False)
            tokenizer = PunktSentenceTokenizer(trainer.get_params())
                       
            #lines = self.booktext.split('\n')
            lines = re.split(r'\n|\[LANGFORMEL\]',self.booktext)
            for line in lines:
                sents = tokenizer.tokenize(line.strip())
                for sent in sents:
                    if self.clean_sent(sent) and classif.identify(sent) == 'de':
                        self.len_extracted += len(sent)
                        #print(sent)
                        sent = re.sub(r' +',' ',sent)
                        self.sentences.append(sent)
        else:
            self.last_fs = ''


filelist = glob.glob("XML/*.xml") 

parser = xml.sax.make_parser() 

for fname in filelist:  
    print(fname)
  
    bkhandler = BookHandler()
    parser.setContentHandler(bkhandler)
    f = codecs.open(fname,"r",'utf-8')
    try:
        parser.parse(f)
    except:
        f.close()
        continue
        
    f.close()

    
    te = TextExtract()
    te.setTextType(bkhandler.get_common_fsizes())
    te.setFootnotes(bkhandler.footnotes)
    parser.setContentHandler(te)
    f = codecs.open(fname,"r",'utf-8')
    parser.parse(f)
    f.close()
    
    print(len(te.sentences),te.len_extracted/bkhandler.total_len)
    
    f_base = os.path.basename(fname)
    
    file_out_txt = 'TXT/'+f_base[:-4] +'.txt'
    fout = codecs.open(file_out_txt,'w','utf8')
    for sent in te.sentences:
        fout.write(sent)
        fout.write('\n')
    fout.close()



