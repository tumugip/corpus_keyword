import pandas as pd
import argparse
import os
from janome.tokenizer import Tokenizer
import datetime


# アーギュメントを設定する
parser = argparse.ArgumentParser()
parser.add_argument('train_file',help='train_file_name')
parser.add_argument('test_file',help='test_file_name')
args = parser.parse_args()

TRAIN_tsv = args.train_file
TEST_tsv = args.test_file
stop_word = ['し','その','する']
t = Tokenizer()


# 書き出しの下準備
today = str(datetime.date.today())
folder_path = './keyword_corpus'+ str(today)
os.mkdir(folder_path)

train_file_name_list = TRAIN_tsv.split('/')
train_file_name  = train_file_name_list[-1]
train_file_name = train_file_name.replace('.tsv', '')

test_file_name_list = TEST_tsv.split('/')
test_file_name  = test_file_name_list[-1]
test_file_name = test_file_name.replace('.tsv', '')

# 関数定義
def keyword(corpus_df):
  new_keyword_corpus = pd.DataFrame()
  for row in corpus_df.itertuples():
    A = []
    for token in t.tokenize(row[1]):
      if token.part_of_speech.split(',')[0] != '助詞':
        if token.surface.isalpha() == True:
          if token.surface.isascii() == False:
            A.append(token.surface)
    A = " ".join(A)
    S = [A,row[2]]
    df_s = pd.DataFrame(S)
    new_keyword_corpus = pd.concat([new_keyword_corpus, df_s.T])
  new_keyword_corpus = new_keyword_corpus.reset_index(drop=True)
  return new_keyword_corpus

def JPmatch(corpus_df):
  for row in corpus_df.itertuples():
    df = corpus_df[int(row[0])+1:]
    for row2 in df.itertuples():
      if row[1] == row2[1]:
        corpus_df = corpus_df.drop(row2[0])
  return corpus_df

def JPPYmatch(corpus_df):
  for row in corpus_df.itertuples():
    df = corpus_df[int(row[0])+1:]
    for row2 in df.itertuples():
      if row[1] == row2[1] and row[2] == row2[2]:
        corpus_df = corpus_df.drop(row2[0])
  return corpus_df

def re_stop_word(corpus_df):
  for row in corpus_df.itertuples():
    JP = row[1].split(' ')
    match = []
    for i in range(len(JP)):
      for j in range(len(stop_word)):
        if JP[i] == stop_word[j]:
          match.append(i)
      # print(match)
    if len(match) > 0:
      for n in range(len(match)):
        del JP[match[n]-n]
    moziretu = ' '.join(JP)
    corpus_df.loc[row[0],0] = moziretu
  return corpus_df


# trainファイル
old_train_corpus = pd.read_csv(TRAIN_tsv,header=None,sep='\t')
old_train_corpus = old_train_corpus.dropna(how='any')


key_train = keyword(old_train_corpus)
key_train=key_train.dropna(how='any')
key_train.to_csv(folder_path+'/'+train_file_name+'_keyword'+today+'.tsv',sep='\t',header=False, index=False)


key_train_JPPYmatch = JPPYmatch(key_train)
key_train_JPPYmatch=key_train_JPPYmatch.dropna(how='any')
key_train_JPPYmatch.to_csv(folder_path+'/'+train_file_name+'_keyword_JPPYmatch'+today+'.tsv',sep='\t',header=False, index=False)


key_train_JPmatch = JPmatch(key_train)
key_train_JPmatch=key_train_JPmatch.dropna(how='any')
key_train_JPmatch.to_csv(folder_path+'/'+train_file_name+'_keyword_JPmatch'+today+'.tsv',sep='\t',header=False, index=False)


key_train_JPPYmatch_stop = re_stop_word(key_train_JPPYmatch)
key_train_JPPYmatch_stop=key_train_JPPYmatch_stop.dropna(how='any')
key_train_JPPYmatch_stop.to_csv(folder_path+'/'+train_file_name+'_keyword_JPPYmatch_stopword'+today+'.tsv',sep='\t',header=False, index=False)


key_train_JPmatch_stop = re_stop_word(key_train_JPmatch)
key_train_JPmatch_stop=key_train_JPmatch_stop.dropna(how='any')
key_train_JPmatch_stop.to_csv(folder_path+'/'+train_file_name+'_keyword_JPmatch_stopword'+today+'.tsv',sep='\t',header=False, index=False)


# testファイル
old_test_corpus = pd.read_csv(TEST_tsv,header=None,sep='\t')
old_test_corpus = old_test_corpus.dropna(how='any')


key_test = keyword(old_test_corpus)
key_test=key_test.dropna(how='any')
key_test.to_csv(folder_path+'/'+test_file_name+'_keyword'+today+'.tsv',sep='\t',header=False, index=False)


key_test_JPPYmatch = JPPYmatch(key_test)
key_test_JPPYmatch=key_test_JPPYmatch.dropna(how='any')
key_test_JPPYmatch.to_csv(folder_path+'/'+test_file_name+'_keyword_JPPYmatch'+today+'.tsv',sep='\t',header=False, index=False)


key_test_JPmatch = JPmatch(key_test)
key_test_JPmatch=key_test_JPmatch.dropna(how='any')
key_test_JPmatch.to_csv(folder_path+'/'+test_file_name+'_keyword_JPmatch'+today+'.tsv',sep='\t',header=False, index=False)


key_test_JPPYmatch_stop = re_stop_word(key_test_JPPYmatch)
key_test_JPPYmatch_stop=key_test_JPPYmatch_stop.dropna(how='any')
key_test_JPPYmatch_stop.to_csv(folder_path+'/'+test_file_name+'_keyword_JPPYmatch_stopword'+today+'.tsv',sep='\t',header=False, index=False)


key_test_JPmatch_stop = re_stop_word(key_test_JPmatch)
key_test_JPmatch_stop=key_test_JPmatch_stop.dropna(how='any')
key_test_JPmatch_stop.to_csv(folder_path+'/'+test_file_name+'_keyword_JPmatch_stopword'+today+'.tsv',sep='\t',header=False, index=False)



