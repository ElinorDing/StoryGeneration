from collections import defaultdict
import os
import re
import wordninja
import contractions
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from collections import Counter
import math
import csv

data_path = '/Users/vonnet/Master/masterProject/Dataset/RawDataset/'
new_data_path = '/Users/vonnet/Master/masterProject/Dataset/FinalDataset/'

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

class Preprocessing:
    def __init__(self):
        self.path = data_path
        self.raw_data = self.read_data()
        self.story_data = self.story_preprocessing()
        self.keyword_data = self.keyword_preprocessing()
        self.top_keywords = self.most_relevant_keywords_data(5)
        self.final_data = self.match_keyword_story()

    # Read each story from the given path
    def read_data(self):
        data = defaultdict(list)
        story_code = 0
        for filename in os.listdir(self.path):
            story_code += 1
            file = os.path.join(self.path, filename)
            # checking if it is a file
            if os.path.isfile(file):
                # filename = self.preprocess_story_title(filename)
                with open(file, 'r', encoding="utf-8") as f:
                    data[story_code] = f.read()
            else:
                raise Exception(file + " is not a file!")
        return data
    
    # Preprocess the story
    def story_preprocessing(self):
        fine_data = defaultdict(list)
        for key, value in self.raw_data.items():
            value = re.sub(r"[^\w\s,.?]", '', contractions.fix(value))
            value = re.sub(r'\d+', '', value)
            special_term = 'The New Social StoryTM Book, th Anniversary Edition  by Carol Gray, Future Horizons, Inc.'
            value = value.replace(special_term, ' ')
            value = value.replace('\n', ' ')
            clean_data = value.split('.')

            clean_data = '. '.join(clean_data)

            split_data = clean_data.split('.')
            split_data = [s.lstrip() for s in split_data if s.strip()] # remove empty string and empty space at the beginning
            # Regular expression pattern to match a string that contains only one 'n' and spaces
            pattern = re.compile(r'^\s*n\s*$')
            # Filter list to remove strings that match the pattern
            filtered_list = [s for s in split_data if not pattern.match(s)]

            list_with_dots = [s + "." for s in filtered_list if not s.endswith('.')]
            clean_data = ' '.join(list_with_dots)

            fine_data[key] = clean_data
        return fine_data
    
    def clean(self, document):
        stop_free = " ".join([word for word in document.lower().split() if word not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized 
    
    # Extract keywords from the story
    def keyword_preprocessing(self):
        token_docs = defaultdict(list)
        tf = defaultdict(list)
        for key, value in self.story_data.items():
            docs = value.split('.')
            docs = [s.lstrip() for s in docs if s.strip()]
            # separate words
            docs = [wordninja.split(i) for i in docs]
            docs = list(map(' '.join, docs))
            # remove stop words, punctuation, and lemmatization
            doc_clean = [self.clean(doc).split() for doc in docs]
            tokens_per_doc = [item for sublist in doc_clean for item in sublist]
            token_docs[key] = tokens_per_doc # all tokens in one story with duplicates

            # LDA
            # # Create a dictionary and corpus
            # dictionary = corpora.Dictionary(doc_clean)
            # corpus = [dictionary.doc2bow(text) for text in doc_clean]
            # # Run LDA
            # lda_model = LdaModel(corpus, num_topics=2, id2word = dictionary, passes=50)
            # print(lda_model.print_topics(num_topics=2, num_words=3))

        # TF
        tf = {}
        for code, tokens_per_doc in token_docs.items():
            tf[code] = dict(Counter(tokens_per_doc))
        
        # DF
        token_collection = {}
        for code, tokens in tf.items():
            token_collection[code] = list(tokens.keys()) # all tokens in one story without duplicates
        all_tokens = [token for tokens in token_collection.values() for token in tokens]
        count_tokens_per_collection = dict(Counter(all_tokens)) 
        print(count_tokens_per_collection) 

        # IDF
        idf = {}
        N = len(token_docs)
        for token, df in count_tokens_per_collection.items():
            idf[token] = math.log10(N/df)

        # TF-IDF
        tf_idf = {}
        for code, content in tf.items():
            tf_idf[code] = {}
            for token, tf in content.items():
                tf_idf[code][token] = tf * idf[token]
        
        for code, token_weight in tf_idf.items():
            sort_tf_idf = sorted(token_weight.items(), key = lambda x:x[1], reverse=True)
            token_weight.clear()
            for k, v in sort_tf_idf:
                tf_idf[code][k] = v

        return tf_idf
    
    # Get the most relevant keywords for each story
    def most_relevant_keywords_data(self, top_n):

        if top_n <= 0:
            raise Exception("The given parameter is invalid!")
        else:
            tf_keywords = {}
            for code, token_weight in self.keyword_data.items():
                list_token = list(token_weight.keys())[:top_n]
                tf_keywords[code] = list_token
            return tf_keywords
    
    # Match the keywords with the corresponding original stories
    def match_keyword_story(self):

        final_data = defaultdict(list)
        for code, tokens in self.top_keywords.items():
            token_string = ', '.join(tokens)
            final_data[token_string] = self.story_data[code]
        return final_data

    def write_csv(self):
        filename = os.path.join(new_data_path, 'dataset.csv')
        with open(filename, 'w') as csvfile:
            fieldnames = ['keywords', 'story']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for keywords, story in self.final_data.items():
                writer.writerow({'keywords': keywords, 'story': story})  
            
if __name__ == '__main__':
    preprocessing = Preprocessing()
    preprocessing.write_csv()