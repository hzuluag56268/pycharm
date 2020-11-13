
Regular expressions & word tokenization

'''Simple topic identification'''


from nltk.tokenize import word_tokenize
from collections import Counter

bow = Counter(word_tokenize
        ("""The cat is in the box. T
        he cat likes the box.      
                    The box is over the cat.""").lower())
bow.most_common(2)
[('The', 3), ('box', 3)


from ntlk.corpus import stopwords
stop_words = stopwords.words('english')

# str.isalpha()

from nltk.stem import WordNetLemmatizer
WordNetLemmatizer.lemmatize(t) # Used in the last comprehension
# list before the bag of words




Introduction to gensim .......


from gensim.corpora.dictionary import Dictionary

tokenized_docs  = [[tokenised words],[tokenised words]]]
dictionary = Dictionary(tokenized_docs)
dictionary.token2id #Method to inverse key value pairs
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
# dictionary.doc2bow() returns list of (int, int)-BoW representation of document.
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count
#itertools.chain.from_iterable() Allows to iterate over list of
# lists as though it were one


from gensim.models.tfidfmodel import TfidfModel
# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus) #the corpus here is list of lists of (int, int)-BoW
#TfidfModel always initialized using the corpus

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[corpus[4]] #corpus[4] is simple list of (int, int)-BoW


'''Named-entity recognition'''

nltk.pos_tag
nltk.ne_chunk
nltk.ne_chunk_sents

pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]
# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1



...Introduction to SpaCy


# Import spacy
import spacy#you also need to download the pre trained models and corpus vectors

# Instantiate the English model: nlp
nlp = spacy.load('en')

# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(': ',ent, '2 ', ent.text)
    print(ent.label_, ent.text)



....Multilingual NER with polyglot

from polyglot.text import Text  ##ou also need to download the pre trained models and corpus vectors

# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
    print(ent, ent.tag)




from sklearn.naive_bayes import MultinomialNB
#for nlp