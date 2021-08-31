import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

def generate_squares(N):
  for my_number in range(N):
    yield my_number ** 2

# # # Eamil bidy extraction # # #

def email_body_generator(path):
  for root, dirnames, filenames, in walk(path):
    for file_name in filenames:
      filepath = join(root, file_name)
      stream = open(filepath, encoding="latin-1")
      # message = stream.read()
      is_body = False
      lines = []
      for line in stream:
        if is_body:
          lines.append(line)
        elif line == "\n":
          is_body = True
      stream.close()
      email_body = "\n".join(lines)
      yield file_name, email_body

def df_from_directory(path, classification):
  rows = []
  row_names = []

  for file_name, email_body in email_body_generator(path):
    rows.append({"MESSAGE": email_body, "CATEGORY": classification})
    row_names.append(file_name)

  return pd.DataFrame(rows, index=row_names)

def clean_message(message, stemmer=PorterStemmer(),
                  stop_words=set(stopwords.words("english"))):
  # # Converts tp Lower Case and splits up the words
  words = word_tokenize(message.lower())

  filtered_words = []
  for word in words:
    # # Remove the stop words and punctuation
    if word not in stop_words and word.isalpha():
      filtered_words.append(stemmer.stem(word))

  return filtered_words


def clean_msg_no_html(message, stemmer=PorterStemmer(),
                  stop_words=set(stopwords.words("english"))):
  # # Remove HTML tags
  soup = BeautifulSoup(message, "html.parser")
  cleaned_text = soup.get_text()

  # # Converts to Lower Case and splits up the words(Tokening)
  words = word_tokenize(cleaned_text.lower())

  filtered_words = []
  for word in words:
    # # Remove the stop words and punctuation
    if word not in stop_words and word.isalpha():
      # # Remove Word Stemming
      filtered_words.append(stemmer.stem(word))
      # filtered_words.append(word)

  return filtered_words

def make_sparse_matrix(df, indexed_words, labels):
    """
    Returns sparse matrix as dataframe.
    
    df: A dataframe with words in the columns with a document id as an index (X_train or X_test)
    indexed_words: index of words ordered by word id
    labels: category as a series (y_train or y_test)
    """
    
    nr_rows = df.shape[0]
    nr_cols = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []
    
    for i in range(nr_rows):
        for j in range(nr_cols):
            
            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels.at[doc_id]
                
                item = {'LABEL': category, 'DOC_ID': doc_id,
                       'OCCURENCE': 1, 'WORD_ID': word_id}
                
                dict_list.append(item)
    
    return pd.DataFrame(dict_list)