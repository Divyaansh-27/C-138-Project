import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
import json
import pickle
import numpy as np
words=[]
classes=[]
word_tag_list=[]
ignore_words=['?','!',',','.',"'s","'m"]
train_data_file=open('intents.json').read()
intents=json.loads(train_data_file)
def get_stem_words(words,ignore_words):
    stem_words=[]
    for word in words:
        if word not in ignore_words:
            w=stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words
for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern_word=nltk.word_tokenize(pattern)
        words.extend(pattern_word)
        word_tag_list.append((pattern_word,intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
        stem_words=get_stem_words(words,ignore_words)
print(stem_words)
print(word_tag_list)
print(classes)
def create_bot_corpus(stem_words,classes):
    stem_words=sorted(list(set(stem_words)))
    classes=sorted(list(set(classes)))
    pickle.dump(stem_words,open('wor.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))
    return stem_words,classes
stem_words,classes=create_bot_corpus(stem_words,classes)
print(stem_words)
print(classes)
print(word_tag_list)
training_data=[]
number_of_tags=len(classes)
print(number_of_tags)
#[0]*3= 0
labels=[0]*number_of_tags
print(labels)
for word_tags in word_tag_list:
    print(word_tags)
    bag_of_words=[]
    pattern_word=word_tags[0]
    print(pattern_word)
    for word in pattern_word:
        print(word)
        index=pattern_word.index(word)
        print(index)
        word=stemmer.stem(word.lower())
        pattern_word[index]=word
    print(pattern_word)
    for i in stem_words:
        if i in pattern_word:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    print(bag_of_words)
    labels_encoding=list(labels)
    tag=word_tags[1]
    print(tag)
    tag_index=classes.index(tag)
    print(tag_index)
    labels_encoding[tag_index]=1
    print(labels_encoding)
    training_data.append([bag_of_words,labels_encoding])
    print(training_data)
def preprocess_train_data(training_data):
    training_data=np.array(training_data,dtype=object)
    train_x=list(training_data[:,0])
    train_y=list(training_data[:,1])
    print(train_x[0])
    print(train_y[0])
    return train_x,train_y
train_x,train_y=preprocess_train_data(training_data)