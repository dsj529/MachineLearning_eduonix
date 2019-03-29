'''
Created on Mar 28, 2019

@author: dajoseph

exploring the capabilities of the NLTK library
'''

import nltk
from nltk.corpus import state_union, words
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

pst = PunktSentenceTokenizer(train_text)
tokenized = pst.tokenize(sample_text)

def process_content():
    ''' tokenize and tag words with parts of speech
    
    then analyze the text in phrases defined by a regex-like grammar
    
    also demonstrate "chinking", the elimination of terms from each phrase
       chinked terms are defined between }{ '''
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
            
            chunkGram = r'''Chunk: {RB.?>*<VB.?>*<NNP>+<NN>?}'''
            ''' chunks the text into phrases of:
                <RB> 0 or more adverbs
                <VB> 0 or more verbs
                <NNP> 1 or more proper nouns
                <NN> 0 or 1 singular noun'''
            
            chinkGram = r'''Chunk: {<.*>+}
                                   }<VB.?|IN|DT|TO>+{'''
            ''' removes one or more verbs, prepositions, determiners, and the word 'to' '''
            chunkParser = nltk.RegexpParser(chunkGram) # can replace with chinkGram here
            chunked = chunkParser.parse(tagged)
            
            # draw a graph of the chunks
            # chunked.draw()
            
            # print the chunked phrases normally
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)
    except Exception as e:
        print(str(e))
            
def process_content2():
    ''' testing the named-entity recognition capabilities'''
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            named_ents = nltk.ne_chunk(tagged, binary=True) 
                # binary=True: something is or is not an entity.
                # binary=False: classify the type of entity being named.
                
    except Exception as e:
        print(str(e))
        

            
            
