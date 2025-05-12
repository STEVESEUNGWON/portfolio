import pandas as pd
import numpy as np

# 데이터 불러오기
data = pd.read_csv(r"C:\ITWILL\3_TextMining\data\mbti_1.csv")
df = data.copy()
data.head()

# 데이터 정보 확인
data.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8675 entries, 0 to 8674
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   type    8675 non-null   object
 1   posts   8675 non-null   object
dtypes: object(2)
memory usage: 135.7+ KB
"""

# 데이터에 null값 확인
data.isnull().any()
"""
type     False
posts    False
dtype: bool
"""

# 데이터 type의 종류
np.unique(np.array(data['type']))
"""
array(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
       'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'],
      dtype=object)
"""

# 타입별 게시글 수
post_type = data.groupby(['type']).count()
post_type
"""
      posts
type       
ENFJ    190
ENFP    675
ENTJ    231
ENTP    685
ESFJ     42
ESFP     48
ESTJ     39
ESTP     89
INFJ   1470
INFP   1832
INTJ   1091
INTP   1304
ISFJ    166
ISFP    271
ISTJ    205
ISTP    337
"""

# mbit 타입별 게시글 수 막대차트
import matplotlib.pyplot as plt
# 차트에서 한글 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'






plt.figure(figsize=(25, 10))
plt.bar(post_type.index, height=post_type['posts'])
plt.xlabel('Personality types', size=34)
plt.ylabel('Number of posted', size=34)
plt.title('Total posts for each personality type', size=28)  # Increased title font size
plt.xticks(fontsize=24)  # Increase the font size of x-axis labels
plt.yticks(fontsize=14)  # Increase the font size of y-axis labels
plt.show()





# mbti 타입별 게시글 수 파이 차트
df['I/E'] = df['type'].str[0]
df['N/S'] = df['type'].str[1]
df['T/F'] = df['type'].str[2]
df['J/P'] = df['type'].str[3]

count_ie = df['I/E'].value_counts()
count_ns = df['N/S'].value_counts()
count_tf = df['T/F'].value_counts()
count_jp = df['J/P'].value_counts()

# Create a pie chart for each component with larger text labels and custom colors
fig, axes = plt.subplots(1, 4, figsize=(25, 10))
ax1, ax2, ax3, ax4 = axes.ravel()

# Set the font size of text labels (5 times larger)
textprops = {'fontsize': 25}

# Define custom colors (orange and blue)
colors = ['#ADD8E6', '#FFA500']



ax1.pie(count_ie, labels=count_ie.index, autopct='%1.1f%%', startangle=30, textprops=textprops, colors=colors)
ax1.set_title('Extrovert vs Introvert', fontsize=40)  # Adjust the title font size

ax2.pie(count_ns, labels=count_ns.index, autopct='%1.1f%%', startangle=30, textprops=textprops, colors=colors)
ax2.set_title('Intuitive vs Observant', fontsize=40)  # Adjust the title font size

ax3.pie(count_tf, labels=count_tf.index, autopct=lambda p: f'{p:.1f}%', startangle=30, textprops=textprops, colors=colors)
ax3.set_title('Thinking vs Feeling', fontsize=40)  # Adjust the title font size

ax4.pie(count_jp, labels=count_jp.index, autopct='%1.1f%%', startangle=30, textprops=textprops, colors=colors)
ax4.set_title('Judging vs Prospecting', fontsize=40)  # Adjust the title font size

plt.tight_layout()
plt.show()







# Top 100 단어 선정   
from collections import Counter 
word = []
for sent in df['posts']:
    for i in sent.split():
        word.append(i)
        
Counter(word).most_common(100)

# 단어 구름 시각화

# 전체 post 단어구름 시각화
from wordcloud import WordCloud
wc = WordCloud(width=1200, height=500, 
                         collocations=False, background_color="white", 
                         colormap="tab20b").generate(" ".join(word))

plt.figure(figsize=(25,10))
# generate word cloud, interpolation 
plt.imshow(wc, interpolation='bilinear')
_ = plt.axis("off")

# mbti 타입별 단어구름 시각화




fig, ax = plt.subplots(len(df['type'].unique()), sharex=True, figsize=(25, 18))  # Increase the figsize
k = 0
for i in df['type'].unique():
    
    df_4 = df[df['type'] == i]
    wordcloud = WordCloud(background_color = 'white',max_words=100, relative_scaling=1, normalize_plurals=False).generate(df_4['posts'].to_string())
    ax[k] = plt.subplot(4, 4, k + 1)
    plt.imshow(wordcloud, interpolation='bilinear')
    ax[k].set_title(i, fontsize=24)
    ax[k].axis("off")
    k += 1

fig.patch.set_facecolor('white')
plt.show()





# 게시글 전처리

import pandas as pd
import string
import re
from nltk.corpus import stopwords


def preprocess_text(text):
    # ||| 로 나뉘어 있는 글 나누기
    text = text.replace('|||', ' ')
    
    # url 주소 삭제  
    text = re.sub(r'https?:\/\/.*?[\s+]', ' ', text)
    # s?는 's' 문자가 선택사항이므로 'http' 및 'https'와 모두 일치함
    
    # 길이가 1~2인 단어들을 정규 표현식을 이용하여 삭제
    text = re.sub(r'\W*\b\w{1,2}\b', '', text)
    
    # 영어가 아닌 문자 공백으로 대체
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # 영문소문자 변경
    text = text.lower()
    
    # Remove punctuation : 특수문자 제거
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    
    # mbti 이름 제거
    mbti_types = ["enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp",
                  "infj", "infp", "intj", "intp", "isfj", "isfp", "istj", "istp"]  
    for mbti_type in mbti_types:
        text = text.replace(mbti_type, ' ')
    
    # 공백 제거
    text = ' '.join(text.split())
    
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
            
    return text
# 전처리 적용
df['posts'] = df['posts'].apply(preprocess_text)


# 단어구름 시각화
# 단어 추출
word = []        
for sent in df['posts']:
    for i in sent.split():
        word.append(i)

# Top 100 단어 선정   
from collections import Counter 

word_count = Counter(word) # (dict, list)
top100_word = word_count.most_common(100) 

top100_word

# 단어구름 시각화
from wordcloud import WordCloud
wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf',
          width=500, height=400,
          max_words=100,max_font_size=150,
          background_color='white')

wc_result = wc.generate_from_frequencies(dict(top100_word))

import matplotlib.pyplot as plt 

plt.imshow(wc_result) # imshow(image)
plt.axis('off') # 축 눈금 감추기 
plt.show()

# 각 유형별 워드 클라우드
fig, ax = plt.subplots(len(df['type'].unique()), sharex=True, figsize=(15,len(df['type'].unique())))
k = 0
for i in df['type'].unique():
    df_4 = df[df['type'] == i]
    wordcloud = WordCloud(max_words=1628,relative_scaling=1,normalize_plurals=False).generate(df_4['posts'].to_string())
    plt.subplot(4,4,k+1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(i)
    ax[k].axis("off")
    k+=1

# 텍스트 데이터 수치로 변경
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

cvect = CountVectorizer()
dtm = cvect.fit_transform(df["posts"])

DTM_array = dtm.toarray()
DTM_array.shape # (8675, 97342)

# 타겟 labelencoder
label_encoder = LabelEncoder()
target = df['type']
target = label_encoder.fit_transform(target)

# train/test split : 훈련셋(70) vs 테스트셋(30)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    DTM_array, target, test_size=0.3)

print((x_train.shape), (y_train.shape), (x_test.shape), (y_test.shape))

# Naive Bayes 분류기
from sklearn.naive_bayes import MultinomialNB # nb model
from sklearn.metrics import accuracy_score 

# 학습 모델 만들기 : 훈련셋 이용
nb = MultinomialNB()
model = nb.fit(X= x_train, y = y_train)

# 학습 model 평가 : 테스트셋 이용
y_pred = model.predict(X = x_test)


# 분류정확도 
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('분류정확도 :', acc) # 분류정확도 : 0.89646561659623514
