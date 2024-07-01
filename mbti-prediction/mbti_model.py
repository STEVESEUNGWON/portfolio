import pandas as pd
import numpy as np

data_path = r"C:\ITWILL\3_TextMining\data\MBTI 500.csv"

# CSV 파일을 DataFrame으로 불러오기
data = pd.read_csv(data_path, encoding='utf-8')

# 데이터에 null 값 확인
data.isnull().any()

# 데이터 크기
data.shape # (8675, 2)

# 데이터 정보
data.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 106067 entries, 0 to 106066
Data columns (total 2 columns):
 #   Column  Non-Null Count   Dtype 
---  ------  --------------   ----- 
 0   posts   106067 non-null  object
 1   type    106067 non-null  object
dtypes: object(2)
memory usage: 1.6+ MB
"""

# 타입의 종류
np.unique(np.array(data['type']))
"""
array(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
       'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'],
      dtype=object)
"""

# 타입별 게시글 수
post_type = data.groupby(['type']).count()
type(post_type) # pandas.core.frame.DataFrame
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

# 데이터 전처리
df = data.copy()

import pandas as pd
import string
import re
from nltk.corpus import stopwords

def preprocess_text(text):
    # ||| 로 나뉘어 있는 글 나누기
    text = text.replace('|||', ' ')
    
    # url 주소 삭제  
    text = re.sub(r'https?:\/\/.*?[\s+]', ' ', text)
    
    # 길이가 1~2인 단어들을 정규 표현식을 이용하여 삭제
    text = re.sub(r'\W*\b\w{1,2}\b', '', text)
    
    # 영어가 아닌 문자 공백으로 대체
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # 영문소문자 변경
    text = text.lower()
    
    # Remove punctuation : 특수문자 제거
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    
    # 공백 제거
    text = ' '.join(text.split())
    
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text

# 전처리 적용
df['posts'] = df['posts'].apply(preprocess_text)

df.info()
df['posts'][0]

# 텍스트 데이터 수치로 변경
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

cvect = CountVectorizer(max_features= 8000)
dtm = cvect.fit_transform(df["posts"])
dtm.shape

DTM_array = dtm.toarray()
DTM_array.shape
# (106067, 8000)

"""
tfidf = TfidfVectorizer(max_features= 8000)
DTM = tfidf.fit_transform(df["posts"])
X = DTM.toarray()
X.shape
"""

# 타겟 전처리
label_encoder = LabelEncoder()
target = df['type']
target.value_counts()
target = label_encoder.fit_transform(target)
target[:10]
type(target)

print('디코딩 원본값:',label_encoder.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) )
"""
디코딩 원본값: ['ENFJ' 'ENFP' 'ENTJ' 'ENTP' 'ESFJ' 'ESFP' 'ESTJ' 'ESTP' 'INFJ' 'INFP'
 'INTJ' 'INTP' 'ISFJ' 'ISFP' 'ISTJ' 'ISTP']
"""

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
y_pred

# 분류정확도 
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('분류정확도 :', acc) 
# 분류정확도 : 0.736557619182301

model.score(x_train, y_train)
model.score(x_test, y_test)

"""
디코딩 원본값: [0'ENFJ' 1'ENFP' 2'ENTJ' 3'ENTP' 4'ESFJ' 5'ESFP' 6'ESTJ' 7'ESTP' 8'INFJ' 9'INFP'
 10'INTJ' 11'INTP' 12'ISFJ' 13'ISFP' 14'ISTJ' 15'ISTP']
"""


def classifiers(texts):
    global model, cvect
    
    DTM_test = cvect.transform([texts])
    X_test = DTM_test.toarray()
     

    y_pred = model.predict(X = X_test)
    if y_pred == 0:
        y_pred_result = 'ENFJ'
    elif y_pred == 1:
        y_pred_result = 'ENFP'
    elif y_pred == 2:
        y_pred_result = 'ENTJ'
    elif y_pred == 3:
        y_pred_result = 'ENTP'
    elif y_pred == 4:
        y_pred_result = 'ESFJ'
    elif y_pred == 5:
        y_pred_result = 'ESFP'
    elif y_pred == 6:
        y_pred_result = 'ESTJ'
    elif y_pred == 7:
        y_pred_result = 'ESTP'
    elif y_pred == 8:
        y_pred_result = 'INFJ'
    elif y_pred == 9:
        y_pred_result = 'INFP'
    elif y_pred == 10:
        y_pred_result = 'INTJ'
    elif y_pred == 11:
        y_pred_result = 'INTP'
    elif y_pred == 12:
        y_pred_result = 'ISFJ'
    elif y_pred == 13:
        y_pred_result = 'ISFP'
    elif y_pred == 14:
        y_pred_result = 'ISTJ'
    elif y_pred == 15:
        y_pred_result = 'ISTP'
    return y_pred_result


my_posts = """  They pursue an obsessive and perfectionist tendency. [Source 1] [Source 2]
Their 3rd function is Ti (Introverted Thinking), which means they tend to outwardly show empathy but internally think critically. Therefore, among the Feeling (F) types, they generally lean more towards the Thinking (T) type. [Source 27] So, outwardly, they may appear as if they have a high Thinking (T) tendency.
Their dominant function is Ni (Introverted Intuition), which means they enjoy finding commonalities between the past and present to predict the future. They have excellent imagination, creativity, and originality, which gives them exceptional insights. As a result, they have extraordinary foresight. [Source 29]
Because they primarily use Ni (Introverted Intuition), they are good at public speaking. Outwardly, they might look like the most Thinking (T) type among the Feeling (F) types. [Source 30]
They have clear personal rules and can come across as stubborn. [Source]
It's a rare personality type globally, especially among men. However, in Korea, it's relatively less rare, and in Japan, it's the third most common type. [Source] [Source 2]
They are sensitive to social injustices and possess a strong sense of ethics. [Source]
They view the world from a different perspective than most people. They don't accept anything at face value. [Source] [Source]
In daily life, INFJ is both conservative and rebellious. [Source]
They are mature yet childlike. [Source]
They are passionate about meaningful goals and purposes. This is also a characteristic of the Intuitive (N) type. They are interested in transcending reality and exploring concepts and ideas. [Source]
They absorb other people's emotions like a sponge, making them vulnerable to empathy burnout. [Source] [Source 35]
All personality types have the potential for empathy burnout, but INFJ experiences the greatest difficulty in daily life. They may not even realize why they suddenly feel such intense emotions, and it may seem as if they cannot escape these emotions. Separating other people's emotions from their own can be extremely difficult for INFJ. [Source 36]
They enjoy stimulating their imagination through creative works. [Source]
They are curious, passionate, and always questioning. They have many unanswered questions in their hearts. [Source]
They are dreamy and enjoy mysterious things. [Source]
They dislike lies and prioritize truth. [Source]
They have a strong interest in psychology. [Source]
Under extreme stress, they may feel the urge to act impulsively. [Source] [Source 2]
They have a great interest in the arts, literature, and music, often leading to a strong sense in these areas. Their interest frequently translates into their actions, resulting in many people with outstanding talents in these fields. [Source]
They tend to overthink. This is an inevitable combination, as their few words, rich imagination (N), emotional nature (F), and meticulousness (J) lead to a desire to make decisions.   """
y_pred_result = classifiers(my_posts)       
y_pred_result    










