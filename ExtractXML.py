from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
import pdfminer
import codecs
import re
import csv
from xml.sax.saxutils import escape
import numpy as np                   
import os                     
import glob
import joblib         
import unicodedata
import html
from collections import Counter


# Global Variables
laparams=LAParams()
setattr(laparams, 'all_texts', True)


# Functions for calculating text features
def isbold(obj):
    isbold = 1
    for element in obj:
        for c in element: 
            if isinstance(c, pdfminer.layout.LTChar):
                font_name = c.fontname.lower()
                if "bold" not in font_name:
                    isbold = 0                            
                    return isbold
    return isbold  

def isitalic(obj):
    isitalic = 1
    for element in obj:  
        for c in element:
            if isinstance(c, pdfminer.layout.LTChar):
                font_name = c.fontname.lower()
                if "italic" not in font_name:
                    isitalic = 0
                    return isitalic    
    return isitalic  


def size(obj):
    sizes = Counter()
    for element in obj:
        if isinstance(element, pdfminer.layout.LTChar):  
            sizes.update([element.size])
        elif isinstance(element, pdfminer.layout.LTTextLineHorizontal):  
            for c in element:
                if isinstance(c, pdfminer.layout.LTChar):   
                    sizes.update([c.size])
    if len(sizes) == 0:
        return 1
    return sizes.most_common(1)[0][0]


class Textbox(object):    
    def __init__(self, lttb, pagenr, pagesize, bbox, isbold, isitalic, size, text):
        self.LTTB_list = [lttb]
        self.pagenr = pagenr
        self.pagesize = pagesize
        self.bbox = bbox #x0, y0, x1, y1 
        self.text = text
        self.margins = [pagesize[3] - bbox[3],bbox[0] - pagesize[0],bbox[1] - pagesize[1],pagesize[2] - bbox[2]]        
        self.box_l = 0
        self.box_r = 0
        self.box_o = 0
        self.box_d = 0        
        self.isbold = isbold
        self.isitalic = isitalic
        self.size = size
        self.area = (bbox[0] - bbox[2]) * (bbox[1] - bbox[3]) / 10000.0
        self.height = (bbox[3] - bbox[1])
        self.width = (bbox[2] - bbox[0])
        self.len = len(text) #original: with double blanks, line breaks,... 
  
    #functions to order Textboxes by their upper left corner
    #these functions are needed to enable sorting of Textboxes
   
    def __eq__(self, other):
        return ((self.pagenr, self.bbox[0], self.bbox[1] ) == (other.pagenr, other.bbox[0], other.bbox[1]))

    def __ne__(self, other):
        return ((self.pagenr, self.bbox[0], self.bbox[1] ) != (other.pagenr, other.bbox[0], other.bbox[1]))

    def __lt__(self, other):
        if self.pagenr < other.pagenr:
            return True
        if self.pagenr > other.pagenr:
            return False
        
        if self.bbox[1] > other.bbox[3]:
            return True
        if self.bbox[3] < other.bbox[1]:
            return False
        if self.bbox[0] < other.bbox[0]:
            return True
        return False
        
    
    def __le__(self, other):
        return (self < other or self == other)
            
    def __gt__(self, other):
        return other < self
    
    def __ge__(self, other):
        return other <= self
            
    def h_overlap(self,other,strict = False):
        if strict:
            return (self.bbox[0] <= other.bbox[0] <= self.bbox[2]) or (self.bbox[0] <= other.bbox[2] <= self.bbox[2])
        if (self.bbox[0] <= other.bbox[0] <= self.bbox[2]) or (self.bbox[0] <= other.bbox[2] <= self.bbox[2]):
            overlap = min(self.bbox[2],other.bbox[2]) - max(self.bbox[0],other.bbox[0])
            if overlap > 0.3 * min(self.bbox[2] - self.bbox[0] , other.bbox[2] - other.bbox[0]):
                return True                  
        return False
                       
    def v_overlap(self,other, strict = False):
        if strict:
            return (self.bbox[1] <= other.bbox[1] <= self.bbox[3]) or (self.bbox[1] <= other.bbox[3] <= self.bbox[3])
        if (self.bbox[0] <= other.bbox[1] <= self.bbox[3]) or (self.bbox[1] <= other.bbox[3] <= self.bbox[3]):
            overlap = min(self.bbox[3],other.bbox[3]) - max(self.bbox[1],other.bbox[1])
            if overlap > 0.3 * min(self.bbox[3] - self.bbox[1] , other.bbox[3] - other.bbox[1]):
                return True
        return False
                       
    
    def get_feats(self): 
        return [self.bbox[0],self.bbox[1],self.bbox[2],self.bbox[3],self.margins[0],self.margins[1],self.margins[2],self.margins[3],self.box_l,self.box_r,self.box_o,self.box_d,self.isbold,self.isitalic,self.size,self.area,self.height,self.width,self.len]
        


# Functions to analyze a PDF document
# Find all text boxes, collect information on these boxes and return a list of complex objects representing the boxes

def getTextDecorated(obj):
    text = ""
    sizes = []
    lb = []
    #ub =[]
    for element in obj:
        for c in element:
            if isinstance(c, pdfminer.layout.LTChar) and not c.get_text().startswith('(cid:'):  
                text += c.get_text()
                sizes.append(c.size)
                lb.append(c.bbox[1])
                #ub.append(c.bbox[3])
            else:
                ct = c.get_text()
                text += ct
                sizes.extend(len(ct)*[0])
                lb.extend(len(ct)*[0])
                #ub.append(0)
    sizecnt = Counter(sizes)
    mainsize = sizecnt.most_common()[0][0]
    
    textdec = ""
    sup = False
    #sub = False
    for i in range(len(sizes)):
        s = sizes[i]
        c = text[i]
        if i > 0 and 0 < s < 0.8*mainsize and c in "0123456789-": #not in ' \t\r\n\.,-=':
            if not sup and lb[i] > lb[i-1]  :
                textdec += '|sup|'
                sup = True
                #sub = False
            #elif not sub and ub[i] < ub[i-1]  :
            #    textdec += '|sub|'
            #    sub = True
            #    sup = False
                
        elif sup:
            textdec += '|/sup|'
            sup = False
        #elif sub:
        #    textdec += '|/sub|'
        #    sub = False
            
        textdec += c
    
    if sup:
        textdec += '|/sup|'
    #elif sub:
    #    textdec += '|/sub|'  
    
    return textdec

def parse_obj(pagenr,pagesize,lt_objs):
    elements = []
    # loop over the object list
    for obj in lt_objs:
        
       # if it's a textbox, print text and location
        if isinstance(obj, pdfminer.layout.LTTextBoxHorizontal):
            #text = obj.get_text()
            text = getTextDecorated(obj)
            elements.append(Textbox(obj,pagenr, pagesize,(obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]) ,isbold(obj), isitalic(obj), size(obj), text))
            
        # if it's a container, recurse
        elif isinstance(obj, pdfminer.layout.LTFigure):
            elements.extend(parse_obj(pagenr,pagesize,obj._objs))
    return elements


def read_pdf(file):
    elements = []
    # Open a PDF file.
    fp = open(file, 'rb')

    # Create a PDF parser object associated with the file object.
    parser = PDFParser(fp)
    document = PDFDocument(parser)

    # Check if the document allows text extraction. If not, abort.
    if not document.is_extractable:
        return [] 
        #raise PDFTextExtractionNotAllowed

    # Create a PDF resource manager object that stores shared resources.
    rsrcmgr = PDFResourceManager()

    # Create a PDF device object.
    device = PDFDevice(rsrcmgr)

    # Create a PDF page aggregator object.
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)

    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
              
    # loop over all pages in the document
    page_nr = 0
    for page in PDFPage.create_pages(document): 
        # read the page into a layout object
        interpreter.process_page(page)

        layout = device.get_result()
        # print(layout) # Seitengröße und Rotationsgrad
        # extract text from this object
        elements.extend(parse_obj(page_nr,page.mediabox,layout._objs))
        #for page.cropbox , mediabox is paper size, cropbox is printable size
        page_nr += 1

    return elements


def neighbor_box(elements):
    for e in elements:
        elements_page = []
        box_o,box_d,box_l,box_r = 0, 0, 0, 0
        
        for e_er in elements:   #Summarize all elements per page         
            if e.pagenr == e_er.pagenr: 
                elements_page.append(e_er)
            else:    
                continue
                    
        for e_page in elements_page: #For all elements on a page check if there are adjacent boxes  
            if e.v_overlap(e_page): 
                if e.bbox[0] < e_page.bbox[0]:
                    e.box_r = 1
                elif e.bbox[0] > e_page.bbox[0]:
                    e.box_l = 1

            if e.h_overlap(e_page): 
                if e.bbox[1] < e_page.bbox[1]:
                    e.box_o = 1
                elif e.bbox[1] > e_page.bbox[1]:
                    e.box_d = 1
            
            elif ((e.bbox[0] >= e_page.bbox[0]) and (e.bbox[2] < e_page.bbox[2])) or ((e.bbox[0] <= e_page.bbox[0]) and (e.bbox[2] > e_page.bbox[2])):
                if e.bbox[1] < e_page.bbox[1]:
                    e.box_o = 1
                elif e.bbox[1] > e_page.bbox[1]:
                    e.box_d = 1     

    return elements

def CleanInvalidXmlChars(text):
    #From xml spec valid chars: 
    # #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]     
    # any Unicode character, excluding the surrogate blocks, FFFE, and FFFF. 

    return re.sub(r"[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]", "",text); 



def write_txt(elements,linedist,file_out_txt,file_in):
    file = codecs.open(file_out_txt,'w+','utf-8')

    print("<extracted source='%s'>"%file_in,file=file)
    pagenr = -1
    page_open = False
    txt_open = False
    
    isbold = False
    isitalic = False
    fontsize = -1
    lowerb = 1000000
    leftb = 0
        
    for e in elements:
        if e.pagenr != pagenr:
            pagenr = e.pagenr
            if txt_open:
                print("</txt>",file=file) 
                txt_open = False
            if page_open:
                print("</page>",file=file)    
            print("<page nr='%d'>"%pagenr,file=file) 
            page_open = True
            isbold = False
            isitalic = False
            fontsize = -1
        margin = abs(lowerb - e.bbox[3])
        offset = abs(leftb - e.bbox[0])
        if margin > 0.2 + linedist.get(fontsize,5.8) or offset > 2 * fontsize or round(e.size,2) != fontsize or e.isitalic != isitalic or e.isbold != isbold:
        #if margin > 0.6 * max(10,fontsize) or offset > 2 * fontsize or round(e.size,2) != fontsize or e.isitalic != isitalic or e.isbold != isbold:
            isbold = e.isbold
            isitalic = e.isitalic
            fontsize = round(e.size,2)
            if txt_open:
                print("</txt>",file=file)   
            print("<txt fontsize='%.2f' italic='%s' bold='%s'>"%(fontsize,str(isitalic),str(isbold)),file=file) 
            txt_open = True
        cleantext = CleanInvalidXmlChars(e.text)
        xmltext = html.escape(cleantext) 
        xmltext = re.sub("\|sup\|","<sup>",xmltext)
        xmltext = re.sub("\|/sup\|","</sup>",xmltext)
        xmltext = re.sub("\|sub\|","<sub>",xmltext)
        xmltext = re.sub("\|/sub\|","</sub>",xmltext)
        file.write(xmltext)
        lowerb = e.bbox[1]
        leftb = e.bbox[0]
    if txt_open:
        print("</txt>",file=file) 
    print("</page>",file=file)
    print("</extracted>",file=file)
    file.close()


def extract_classify_contract(file_in):
    elements = read_pdf(file_in)
    elements = sorted(elements)
    elements = neighbor_box(elements) 

    return elements


def linedistance(elements):
    linemargins = {}
    prev_bottom = 0
    prev_fontsize = 0
    fontsize = 0
    bottom = 0
    top = 0
    
    for e in elements:
        lttbox = e.LTTB_list[0]
        if isinstance(lttbox, pdfminer.layout.LTTextBoxHorizontal):
            for line in lttbox:
                if isinstance(line, pdfminer.layout.LTTextLineHorizontal):
                    bottom = line.bbox[1]
                    top = line.bbox[3]
                    fontsize = round(size(line),2)
                    m = round(prev_bottom - top,1)
                    #print(prev_bottom, top, bottom)
                    if 0 < m < 20 and prev_fontsize == fontsize:
                        lm_font = linemargins.get(fontsize,Counter())
                        lm_font.update([m])
                        linemargins[fontsize] = lm_font
                    prev_bottom = bottom
                    prev_fontsize = fontsize
        
    linemargins = {fs:cnt.most_common()[0][0] for fs,cnt in linemargins.items()}    
    return linemargins
    

filelist = glob.glob("PDF/*.pdf") 

for f in filelist:  
    print(f)
    elements = extract_classify_contract(f) 
    ld = linedistance(elements)

    f_base = os.path.basename(f)
    file = f_base[:-4]
    
    file_out_txt = 'XML/'+file +'.xml'
    write_txt(elements,ld,file_out_txt,f)




