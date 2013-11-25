import nltk
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
from nltk import sent_tokenize, word_tokenize, pos_tag, NaiveBayesClassifier
from nltk.corpus import wordnet as wn
import nltk.classify.util
from nltk.corpus import movie_reviews

import string

# create our little application :)
app = Flask(__name__)

# Load default config and override config from an environment variable
app.config.update(dict(
    DEBUG=True,
    SECRET_KEY='key'
))

def word_feats(words):
    return dict([(word, True) for word in words])
    
@app.route('/test', methods=['GET', 'POST'])
def test():
    error = None
    if request.method == 'POST':
    	
    	'''initialize temp objects for tested string, 
    	modified string, tokens(sentence) and wtokens(word)'''
    	teststring = ""
    	modif = ""
    	tokens = []
    	wtokens = []
    	
    	#classifier training
    	negids = movie_reviews.fileids('neg')
        posids = movie_reviews.fileids('pos')
        
        negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
        posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
        
        negcutoff = len(negfeats)*3/4
        poscutoff = len(posfeats)*3/4
        
        trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
        classifier = NaiveBayesClassifier.train(trainfeats)
        
        
    	#import test unicode and convert to string
    	teststring = str(request.form['string'])
    	
    	#tokenize paragraphs or sentences
    	tokens = sent_tokenize(teststring)
    	if len(tokens) > 1:
    		for i in range(len(tokens)):
    			wtokens.append(word_tokenize(tokens[i]))
    			out = classifier.classify(word_feats(tokens[i]))
    			flash("Sentence "+str(i+1))
    			flash(out)
    	else:
    		wtokens.append(word_tokenize(tokens[0]))
    		out = classifier.classify(word_feats(wtokens[0]))
    		flash("Sentence 1")
    		flash(out)
    		
    		
    	#output of paragraph stats
        flash("Input String: "+teststring)
        flash("String size: "+str(len(teststring)))
        flash("Sentence Number: "+str(len(tokens)))

        #POS output of the paragraph
        for j in range(len(wtokens)):
        	flash(wtokens)
        	flash("Sentence "+str(j+1)+" POS:")
        	flash(pos_tag(wtokens[j]))
        
        modif = teststring.lower()
        temp = modif
        modif = temp.translate(string.maketrans("",""), string.punctuation)
        tokens = word_tokenize(modif)
        tokens = sorted(set(tokens))
        
        #synset generation
        flash("Synsets:")
        for word in tokens:
        	synsets = wn.synsets(word)
        	flash(word.upper())
        	if len(synsets) > 0:
        		for l in range(len(synsets)):
        			flash(str(l+1)+") "+str(synsets[l].definition))
        	else:
        		flash("Synsets not available.")
    
        
    return render_template('test.html', error=error)


if __name__ == '__main__':
    app.run()