from os import walk
from os.path import join
from re import A
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from matplotlib import cm

from sklearn.model_selection import train_test_split
import functions
import time
SPMA_CAR = 1
HAM_CAT = 0
VOCAB_SIZE = 2500

start = time.time() ##計時器
example_file = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/practice_email.txt"
SPAM_1_PATH = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/spam_assassin_corpus/spam_1"
SPAM_2_PATH = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/spam_assassin_corpus/spam_2"
EASY_NONSPAM_1_PATH = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/spam_assassin_corpus/easy_ham_1"
EASY_NONSPAM_2_PATH = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/spam_assassin_corpus/easy_ham_2"
DATA_JSON_FILE = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/email-text-data.json"
WHALE_FILE = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/wordcloud_resources/whale-icon.png"

DATA_JSON_FILE = 'D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/email-text-data.json'
WORD_ID_FILE = 'D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/word-by-id.csv'

TRAINING_DATA_FILE = 'D:/Code_practice/Data_science_ML/D6/SpamData/02_Training/train-data.txt'
TEST_DATA_FILE = 'D:/Code_practice/Data_science_ML/D6/SpamData/02_Training/test-data.txt'

# #　Open - Close
stream = open(example_file, encoding="latin-1")
# message = stream.read()
is_body = False
lines = []
for line in stream:
  if is_body:
    lines.append(line)
  elif line == "\n":
    is_body = True
stream.close()
# print(type(message))
# print(message)
# print(lines)
email_body = "\n".join(lines)
# print(email_body)

SPAM_CAT = 1
HAM_CAT = 0
spam_emails = functions.df_from_directory(SPAM_1_PATH, SPAM_CAT)   ### category = 1
spam_emails = spam_emails.append(functions.df_from_directory(SPAM_2_PATH, SPAM_CAT)) 
ham_emails = functions.df_from_directory(EASY_NONSPAM_1_PATH, HAM_CAT)
ham_emails = ham_emails.append(functions.df_from_directory(EASY_NONSPAM_2_PATH, HAM_CAT))
# print(spam_emails.head())
# print(spam_emails.shape)
# print(ham_emails.shape)

data = pd.concat([spam_emails, ham_emails])
# print("Shape of entire dataframe is", data.shape)
# print(data.head())
# print(data.tail())

# # # # Data Clearning: Checking for Missing Value # # # #
## Check if any message bodies are null
# print(data["MESSAGE"].isnull().values.any())
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # check if there are empty emails (string length zero) # # # # 
# print((data.MESSAGE.str.len() == 0).any())
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# print((data.MESSAGE.str.len() == 0).sum())
# print(data.MESSAGE.isnull().sum())

# # # Locate empty emails # # # Drop undesirable data
# print(type(data.MESSAGE.str.len() == 0))
# print(data[data.MESSAGE.str.len() == 0].index)

# print(data.index.get_loc("cmds"))

# data = data.drop(["cmds"])
data.drop(["cmds"], inplace=True)
# print("Shape of entire dataframe is", data.shape)

### Add Document IDs to Track Emails in Dataset
document_ids = range(0, len(data.index))
# print(document_ids)
data["DOC_ID"] = document_ids
data["FILE_NAME"] = data.index
data.set_index("DOC_ID", inplace=True)
# print(data.head())

### Save to File using Pandas
data.to_json(DATA_JSON_FILE)
print(data.CATEGORY.value_counts())

amount_of_spam = data.CATEGORY.value_counts()[1]
amount_of_ham = data.CATEGORY.value_counts()[0]
## mehtod 1
category_names = ["Spam", "Legit Mail"]
sizes = [amount_of_spam, amount_of_ham]
custom_colors = ["#ff7675","#74b9ff"]
plt.figure(figsize=(2.3,2), dpi=240)
plt.pie(sizes, labels=category_names, textprops={"fontsize":6},startangle=90,
        autopct="%1.1f%%", colors=custom_colors, explode=[0, 0.05])
# plt.show()
## method 2
category_names = ["Spam", "Legit Mail"]
sizes = [amount_of_spam, amount_of_ham]
custom_colors = ["#ff7675","#74b9ff"]
plt.figure(figsize=(2.3,2), dpi=240)
plt.pie(sizes, labels=category_names, textprops={"fontsize":6},startangle=90,
        autopct="%1.1f%%", colors=custom_colors)
# draw circle
centre_circle = plt.Circle((0, 0), radius=0.6, fc="white")
plt.gca().add_artist(centre_circle)
# plt.show()
## method 3
category_names = ["Spam", "Legit Mail", "Updates", "Promptions"]
sizes = [25, 43, 19, 22]
custom_colors = ["#ff7675","#74b9ff", "#e55039", "#38ada9"]
plt.figure(figsize=(2.3,2), dpi=240)
plt.pie(sizes, labels=category_names, textprops={"fontsize":6},startangle=90,
        autopct="%1.1f%%", colors=custom_colors, pctdistance=0.8)
# draw circle
centre_circle = plt.Circle((0, 0), radius=0.6, fc="white")
plt.gca().add_artist(centre_circle)
# plt.show()



clean_result = functions.clean_msg_no_html(email_body)
# print(clean_result)

# # clean_message
'''all data -- message'''
nested_list = data.MESSAGE.apply(functions.clean_msg_no_html)
# # print(nested_list)
# # flat_list = [item for sublist in nested_list for item in sublist]
# # print(len(flat_list))
# # print(flat_list)
# # print(nested_list.tail())
# # print(data.head())
'''儲存各自的index(編號)'''
doc_ids_spam = data[data.CATEGORY == 1].index
# # print(doc_ids_spam)
doc_ids_ham = data[data.CATEGORY == 0].index
# # print(doc_ids_ham)
nested_list_ham = nested_list.loc[doc_ids_ham]
nested_list_spam = nested_list.loc[doc_ids_spam]
# # # print(nested_list_ham.shape)

'''將所有email的'字'，串起來變成list(ham)'''
flat_list_ham = [item for sublist in nested_list_ham for item in sublist]
'''排列ham的 unique(已計算重複的)多少"字"，排成每一封信一條的series格式'''
normal_words = pd.Series(flat_list_ham).value_counts()
print(normal_words.shape[0]) # total number of unique words in the non-spam messages
print(normal_words[:10])

flat_list_spam = [item for sublist in nested_list_spam for item in sublist]
spammy_words = pd.Series(flat_list_spam).value_counts()
print(spammy_words.shape[0]) # total number of unique words in the spam messages
'''WordCloud'''
word_cloud = WordCloud().generate(email_body)
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis("off")
# plt.show()

# # download datasets
nltk.download("gutenberg")
nltk.download("shakespeare")
# melville-moby_dick
example_corpus = nltk.corpus.gutenberg.words("melville-moby_dick.txt")
print(len(example_corpus))
print(type(example_corpus))
word_list = [''.join(word) for word in example_corpus]
print(word_list[:10])
novel_as_string = ' '.join(word_list)
print(novel_as_string[:100])
##DATA_JSON_FILE = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/wordcloud_resources/whale-icon.png"
##from PIL import Image
icon = Image.open(WHALE_FILE)
image_mask = Image.new(mode="RGB", size=icon.size, color=(255, 255, 255))
image_mask.paste(icon, box=icon) ## 取出blank canvas
rgb_array = np.array(image_mask)  ## convert the image object to an array 
'''rgb_array.shape = image dimensions'''
word_cloud = WordCloud(mask=rgb_array, 
                        background_color="white",
                        max_words=400)
word_cloud.generate(novel_as_string)
plt.figure(figsize=[10,5])
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis("off")
# plt.show()

hamlet_corpus = nltk.corpus.gutenberg.words("shakespeare-hamlet.txt")
hamlet_word_list = [''.join(word) for word in hamlet_corpus]
hamlet_novel_as_string = ' '.join(hamlet_word_list)
SKULL_FILE = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/wordcloud_resources/skull-icon.png"
skull_icon = Image.open(SKULL_FILE)
skull_mask = Image.new(mode="RGB", size=skull_icon.size, color=(255, 255, 255))
skull_mask.paste(skull_icon, box=skull_icon)
skull_rgb_array = np.array(skull_mask)
skull_hamlet_wordcloud = WordCloud(mask=skull_rgb_array, 
                                    background_color="white",
                                    colormap="bone",
                                    max_words=400)
skull_hamlet_wordcloud.generate(hamlet_novel_as_string)
plt.figure(figsize=[7,7])
plt.imshow(skull_hamlet_wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.show()

### Ham and Spam message ###
THUMBS_UP_FILE = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/wordcloud_resources/thumbs-up.png"
THUMBS_DOWN_FILE = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/wordcloud_resources/thumbs-down.png"
CUSTOM_FONT_FILE = "D:/Code_practice/Data_science_ML/D6/SpamData/01_Processing/wordcloud_resources/OpenSansCondensed-Bold.ttf"
up_icon = Image.open(THUMBS_UP_FILE)
up_mask = Image.new(mode="RGB", size=up_icon.size, color=(255, 255, 255))
up_mask.paste(up_icon, box=up_icon)
up_rgb_array = np.array(up_mask)

down_icon = Image.open(THUMBS_DOWN_FILE)
down_mask = Image.new(mode="RGB", size=down_icon.size, color=(255, 255, 255))
down_mask.paste(down_icon, box=down_icon)
down_rgb_array = np.array(down_mask)

corpus = data.MESSAGE.apply(clean_msg_no_html)

doc_ids_ham = data[data.CATEGORY == 0].index
nested_list_ham = corpus.loc[doc_ids_ham]
word_list_ham = [''.join(word) for sublist in nested_list_ham for word in sublist]   ##flat
# print("\n\nlist\n",word_list_ham[:15])
word_list_ham_as_string = ' '.join(word_list_ham)
ham_wordcloud = WordCloud(mask=up_rgb_array, 
                          background_color="white",
                          colormap="ocean",
                          max_words=500,
                          max_font_size=300,
                          font_path=CUSTOM_FONT_FILE)
ham_wordcloud.generate(word_list_ham_as_string)
plt.figure(figsize=[7,7])
# plt.subplot(1,2,1)
plt.imshow(ham_wordcloud, interpolation="bilinear")
plt.axis('off')
doc_ids_spam = data[data.CATEGORY == 1].index
nested_list_spam = corpus.loc[doc_ids_spam]
word_list_spam = [''.join(word) for word in nested_list_spam]
word_list_spam_as_string = ' '.join(word_list_spam)
spam_wordcloud = WordCloud(mask=down_rgb_array,
                            background_color="white",
                            colormap="gist_heat",
                            max_words=500,
                            max_font_size=300,
                            font_path=CUSTOM_FONT_FILE)
spam_wordcloud.generate(word_list_ham_as_string)
plt.figure(figsize=[7,7])
# plt.subplot(1,2,2)
plt.imshow(spam_wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.show()

# Generate Vocabulary & Dictionary
''' corpus = stemmed_nested_list'''
'''flat_stemmed_list = no category split list'''
flat_stemmed_list = [item for sublist in corpus for item in sublist]
# print(len(flat_stemmed_list))
''' get numbers of each unique word'''
unique_words = pd.Series(flat_stemmed_list).value_counts()
print("Nr of unique words", unique_words.shape[0])
# print(unique_words.head())

# ''' Top frequency words '''
frequent_words = unique_words[0:VOCAB_SIZE]
# # print("Most common words: \n", frequent_words[:10])
# ## Create Vocabulary DataFrame with a WORD_ID
word_ids = list(range(0, VOCAB_SIZE))
vocab = pd.DataFrame({'VOCAB_WORD': frequent_words.index.values}, index=word_ids)
vocab.index.name = "WORD_ID"
# print(vocab.head())

### Saving ###
## Save the Vocabulary as a CSV File
## WORD_ID_FILE = 'SpamData/01_Processing/word-by-id.csv'
vocab.to_csv(WORD_ID_FILE, index_label=vocab.index.name, header=vocab.VOCAB_WORD.name)

clean_email_lengths = [len(sublist) for sublist in corpus]
print('Nr words in the longest email:', max(clean_email_lengths))
print('Email position in the list (and the data dataframe)', np.argmax(clean_email_lengths))

''''''''' Generate Features & a Sparse Matrix '''''''''
### Creating a DataFrame with one Word per Column
# print(type(corpus))
# print(type(corpus.tolist()))
'''to list'''
word_columns_df = pd.DataFrame.from_records(corpus.tolist())
# print(word_columns_df.head())
# print(word_columns_df.shape)

''' Splitting the Data into a Training and Testing Dataset '''
X_train, X_test, y_train, y_test = train_test_split(word_columns_df, data.CATEGORY,
                                                   test_size=0.3, random_state=42)
# print('Nr of training samples', X_train.shape[0])
# print('Fraction of training set', X_train.shape[0] / word_columns_df.shape[0])
X_train.index.name = X_test.index.name = 'DOC_ID'
# X_train.head()
# y_train.head()

''' Create a Sparse Matrix for the Training Data '''
word_index = pd.Index(vocab.VOCAB_WORD)
# print(type(word_index[3]))
# print(word_index.get_loc('thu'))
print("word_index \n",word_index)


sparse_train_df = functions.make_sparse_matrix(X_train, word_index, y_train)
print("sparse_train_df \n",sparse_train_df[:5])


end = time.time()
print("Running time: %f s" % (end - start))
plt.show()