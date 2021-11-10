


import json
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
from wordcloud import WordCloud 
from textblob import TextBlob



apikey = "enter your api key"
url = "enter your url"
authenticator = IAMAuthenticator(apikey)
service = SpeechToTextV1(authenticator = authenticator)
service.set_service_url(url)


with open(join(dirname('__file__'), r'enter your voice path'),'rb') as audio_file:
    dic = json.loads(json.dumps(service.recognize(audio=audio_file,content_type='audio/mp3',model='en-US_NarrowbandModel').get_result(), indent=2))
  



str = ""
while bool(dic.get('results')):
    str = dic.get('results').pop().get('alternatives').pop().get('transcript')+str[:]





text = str

blob = TextBlob(text)
polarity = float(blob.polarity)
subjectivity = float(blob.subjectivity)

polar = ("polarity: " + " %f " ) % float(polarity)
subjective = ("subjectivity: " + " %f " ) % float(subjectivity)

wordcloud = WordCloud().generate(text)
plt.figure(figsize=(20, 16), dpi=30)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("name of image      " + polar +"   "+ subjective,fontsize = 30)

plt.savefig("path.png")
plt.show()
