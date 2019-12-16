import nltk
from nltk.tag import brill
from nltk.corpus import indian
nltk.corpus.indian.words('hindi.pos') 
from nltk.tag import untag
word_patterns = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'), 
    (r'.*देश$', ' NN'),
    (r'.*है$', ' VAUX'),         
    (r'.*सबसे$', ' JJS'),
    (r'.*और$', ' CC'),
    (r'.*ड़ा$', ' JJ'),
    (r'.*वो$', ' PRP'),
    (r'.*रा$', ' PP'),
    (r'.*लो$', ' VB'),
    (r'.*ता$', ' VM'),
    (r'.*गे$', '  VB'),
    (r'.*ना$', ' VB'),
    (r'.*था$', ' VBD'),
    (r'.*थे$', ' VBD'),
    (r'.*कर$', '  VB'),
    (r'.*ह$', '  PRP'),
    (r'.*के$', '  VB'),
    (r'.*आॅं$', ' NN'),         
    (r'.*का$', ' NN'),
    (r'.*हाॅं$', ' PREP'),
    (r'.*दार$', ' JJ'),
    (r'.*आ$', ' JJ'),
    (r'.*धर$', ' PREP'),
    (r'.*म$', ' NNC')
    ]
regexp_tagger = nltk.RegexpTagger(word_patterns)
sentences = list(indian.tagged_sents('hindi.pos'))
print(len(sentences))
training_data = sentences[:50]
gold_data = sentences[50:57] 
testing_data = [untag(s) for s in gold_data]
training_data [0]
brill.Template._cleartemplates()
templates = brill.fntbl37()

trainer = nltk.tag.brill_trainer.BrillTaggerTrainer(initial_tagger=regexp_tagger,templates=templates, trace=3,deterministic=True)
tagger1 = trainer.train(training_data, max_rules=10000)
Rules = tagger1.rules()
text = " वह खाना खाता है ।"
model = tagger1
new_tagged = (model.tag(nltk.word_tokenize(text)))
print(new_tagged)
print()
