# SentEx
Extract lists of sentences from PDF files

This project was developed in the JuVer project to extract text from scanned PDF contracts. The preent version was adopted for the LernMINT project to build a corpus of German text from a large number of scanned books in PDF format.

The program assumes that there is a subdirectory called PDF/ where alle PDF Files are stored. The script ExtractXML.py uses PDFMiner to extract text boxes and recognized text as well as some font and position information of each box. These information is stored in XML-Files in a directory named XML.

The script ExtractText.py extracts sentences from the XML fules and stores these in text files in the subdirectory TXT. Each file will consist of a lists of sentences, where each line is a sentence. This script uses a number of heuristics to remove line breaks and reconstruct hyphenated words. Most of these heuristics are specific for German and might not give usfull results for other languages. The script tries to extract only complete sentences form the main text. Thus footnotes, formula, tables, captiosn etc. most likely are filtered out. Furthermore, only  extracted sentences that are likele to represent complete sentences, starting with a capital and ending with a full stop are stored.

The repository also contains a simple binary classifier that predicts whether a sentence is German or English. This classifier uses n-grams (with n is 1 to 4) as features and was trained on a large number of sentences from German and English Wikipedia pages on physics. Thus it might work best for texts from the physics domain. Only sentences clasified as German are stored. 
