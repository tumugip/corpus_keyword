import pandas as pd
import argparse
import os
from janome.tokenizer import Tokenizer
import datetime



parser = argparse.ArgumentParser()
parser.add_argument('train_file',help='train_file_name')
parser.add_argument('test_file',help='test_file_name')
args = parser.parse_args()



TRAIN_tsv = args.train_file
TEST_tsv = args.test_file
stop_word = ['し','その','する']


# キーワード化

t = Tokenizer()

old_train_corpus = pd.read_csv(TRAIN_tsv,header=None,sep='\t')
new_train_keyword_corpus = pd.DataFrame()

for row in old_train_corpus.itertuples():
  A = []
  for token in t.tokenize(row[1]):
    if token.part_of_speech.split(',')[0] != '助詞':
      if token.surface.isalpha() == True:
        if token.surface.isascii() == False:
          A.append(token.surface)
  A = " ".join(A)
  S = [A,row[2]]
  df_s = pd.DataFrame(S)
  new_train_keyword_corpus = pd.concat([new_train_keyword_corpus, df_s.T])
new_train_keyword_corpus = new_train_keyword_corpus.reset_index(drop=True)


old_test_corpus = pd.read_csv(TEST_tsv,header=None,sep='\t')
new_test_keyword_corpus = pd.DataFrame()

for row in old_test_corpus.itertuples():
  A = []
  B = str(row[1])
  for token in t.tokenize(B):
    if token.part_of_speech.split(',')[0] != '助詞':
      if token.surface.isalpha() == True:
        if token.surface.isascii() == False:
          A.append(token.surface)
  A = " ".join(A)
  S = [A,row[2]]
  df_s = pd.DataFrame(S)
  new_test_keyword_corpus = pd.concat([new_test_keyword_corpus, df_s.T])
new_test_keyword_corpus = new_test_keyword_corpus.reset_index(drop=True)


# コーパスをきれいにする

key_train = new_train_keyword_corpus
key_test = new_test_keyword_corpus

# 空白行を取り除く
key_train=key_train.dropna(how='any')
key_test=key_test.dropna(how='any')

# 日本語-コードの一致行を取り除く
key_train_JPPYmatch = key_train
for row in key_train_JPPYmatch.itertuples():
  df = key_train_JPPYmatch[int(row[0])+1:]
  for row2 in df.itertuples():
    if row[1] == row2[1] and row[2] == row2[2]:
      key_train_JPPYmatch = key_train_JPPYmatch.drop(row2[0])

key_test_JPPYmatch = key_test
for row in key_test_JPPYmatch.itertuples():
  df = key_test_JPPYmatch[int(row[0])+1:]
  for row2 in df.itertuples():
    if row[1] == row2[1] and row[2] == row2[2]:
      key_test_JPPYmatch = key_test_JPPYmatch.drop(row2[0])

# 日本語の一致行を取り除く
key_train_JPmatch = key_train
for row in key_train_JPmatch.itertuples():
  df = key_train_JPmatch[int(row[0])+1:]
  for row2 in df.itertuples():
    if row[1] == row2[1]:
      key_train_JPmatch = key_train_JPmatch.drop(row2[0])
      

key_test_JPmatch = key_test
for row in key_test_JPmatch.itertuples():
  df = key_test_JPmatch[int(row[0])+1:]
  for row2 in df.itertuples():
    if row[1] == row2[1]:
      key_test_JPmatch = key_test_JPmatch.drop(row2[0])


# 日本語一致取り除きデータのストップワードの除去
key_test_JPmatch_stop = key_test_JPmatch
for row in key_test_JPmatch_stop.itertuples():
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
  key_test_JPmatch_stop.loc[row[0],0] = moziretu


key_train_JPmatch_stop = key_train_JPmatch
for row in key_train_JPmatch_stop.itertuples():
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
  key_train_JPmatch_stop.loc[row[0],0] = moziretu


# 日本語-コード一致取り除きデータからストップワード除去
key_test_JPPYmatch_stop = key_test_JPPYmatch
for row in key_test_JPPYmatch_stop.itertuples():
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
  key_test_JPPYmatch_stop.loc[row[0],0] = moziretu


key_train_JPPYmatch_stop = key_train_JPPYmatch
for row in key_train_JPPYmatch_stop.itertuples():
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
  key_train_JPPYmatch_stop.loc[row[0],0] = moziretu


# 再度空白を除去

key_test_JPmatch=key_test_JPmatch.dropna(how='any')
key_train_JPmatch=key_train_JPmatch.dropna(how='any')
key_test_JPPYmatch=key_test_JPPYmatch.dropna(how='any')
key_train_JPPYmatch=key_train_JPPYmatch.dropna(how='any')
key_test_JPmatch_stop=key_test_JPmatch_stop.dropna(how='any')
key_train_JPmatch_stop=key_train_JPmatch_stop.dropna(how='any')
key_test_JPPYmatch_stop=key_test_JPPYmatch_stop.dropna(how='any')
key_train_JPPYmatch_stop=key_train_JPPYmatch_stop.dropna(how='any')



# 書き出しの下準備

folder_path = './keyword_corpus'
os.mkder(folder_path)

today = str(datetime.date.today())

train_file_name = TRAIN_tsv.split('/')
train_file_name_list = TRAIN_tsv.split('/')
train_file_name  = train_file_name_list[-1]
train_file_name = train_file_name.strip('.tsv')


test_file_name = TEST_tsv.split('/')
test_file_name_list = TEST_tsv.split('/')
test_file_name  = test_file_name_list[-1]
test_file_name = test_file_name.strip('.tsv')

# ファイルにする

# 空白を除いたキーワード化
key_train.to_csv(folder_path+train_file_name+'_keyword'+today+'.tsv',sep='\t',header=False, index=False)
key_test.to_csv(folder_path+test_file_name+'_keyword'+today+'.tsv',sep='\t',header=False, index=False)

# 日本語が一致したものを除いたもの
key_train_JPmatch.to_csv(folder_path+train_file_name+'_keyword_JPmatch'+today+'.tsv',sep='\t',header=False, index=False)
key_test_JPmatch.to_csv(folder_path+test_file_name+'_keyword_JPmatch'+today+'.tsv',sep='\t',header=False, index=False)


# 日本語-コードが一致したものを除いたもの
key_train_JPPYmatch.to_csv(folder_path+train_file_name+'_keyword_JPPYmatch'+today+'.tsv',sep='\t',header=False, index=False)
key_test_JPPYmatch.to_csv(folder_path+test_file_name+'_keyword_JPPYmatch'+today+'.tsv',sep='\t',header=False, index=False)


# 日本語が一致したもの、ストップワードを除いたもの
key_train_JPmatch_stop.to_csv(folder_path+train_file_name+'_keyword_JPmatch_stopword'+today+'.tsv',sep='\t',header=False, index=False)
key_test_JPmatch_stop.to_csv(folder_path+test_file_name+'_keyword_JPmatch_stopword'+today+'.tsv',sep='\t',header=False, index=False)


# 日本語-コードが一致したもの、ストップワードをを除いたもの
key_train_JPPYmatch_stop.to_csv(folder_path+train_file_name+'_keyword_JPPYmatch_stopword'+today+'.tsv',sep='\t',header=False, index=False)
key_test_JPPYmatch_stop.to_csv(folder_path+test_file_name+'_keyword_JPPYmatch_stopword'+today+'.tsv',sep='\t',header=False, index=False)
