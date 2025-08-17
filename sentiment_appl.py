#This program scrapes text from four AAPL quarterly reports and performs sentiment analysis using NLTK's SentimentIntensityAnalyzer. 
# The sentiment of the text is measured by positive, negative, or neutral scores in addition to a compound score from -1 to +1. 
# I then compare the compound scores to historical AAPL price data found on Yahoo with yfinance in a plot.

import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import pandas as pd
import datefinder 
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet

quarter = ['first','second','third','fourth']
month = ['02', '05', '08','10']

data = []
scores = []
dates = []
for releases in range(len(quarter)):
    text = " "
    url = 'https://www.apple.com/newsroom/2024/{a}/apple-reports-{b}-quarter-results/'.format(a=month[releases], b=quarter[releases])
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(html, 'html.parser')
    text_all = soup.find_all(attrs={"class": "pagebody-copy"})
    #print(text_all) # This shows all the text we gathered from the Apple website about their quarterly reports in a HTML format.

    for element in text_all:
        text_0 = element.get_text(" ", strip=True) # the get_text function removes all the HTML tags and gives us the clean text
        #print(text_0) # All the text from the text_all part, without the HTML tags
        text = text + text_0

    # This part cleans the text to use in the future
    sent = sent_tokenize(text)
    #print(sent)
    words = [word_tokenize(t) for t in sent]
    list_words = sum(words,[])
    lower_words = [w.lower() for w in list_words] # lowers the whole text to lowercase
    remove_words = [w for w in lower_words if w not in stopwords.words('english')] # Remove the stopwords (eg. and, is, a)
    punctuation_words = [w for w in remove_words if w.isalnum()] # Remove any punctuations that might be there in the text

    def get_wordnet_pos(word):
        '''Map POS tag to first character lemmatize() accepts'''
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {'J': wordnet.ADJ,
                    'N': wordnet.NOUN,
                    'V': wordnet.VERB,
                    'R': wordnet.ADV}
        
        return tag_dict.get(tag, wordnet.NOUN)
    
    final_words = [WordNetLemmatizer().lemmatize(w, get_wordnet_pos(w)) for w in punctuation_words]
    delete_list = ['cupertino','california'] # This is the address of Apple, not needed for us
    words_v2 = [w for w in final_words if w not in delete_list]
    unique_string = (" ").join(words_v2)
    #print(unique_string)

    # Use datefinder to find first match in text. Then break because date always at beginning of quarterly release
    matches = datefinder.find_dates(unique_string)
    for match in matches:
        #print(match)
        break
    dates.append(match)
    # Output:
    # 2023-12-30 00:00:00
    # 2024-03-30 00:00:00
    # 2024-06-29 00:00:00
    # 2024-09-28 00:00:00

    sia = SentimentIntensityAnalyzer()
    #print(sia.polarity_scores(unique_string))
    # Output:
    # {'neg': 0.0, 'neu': 0.83, 'pos': 0.17, 'compound': 0.9803}
    # {'neg': 0.0, 'neu': 0.801, 'pos': 0.199, 'compound': 0.989}
    # {'neg': 0.0, 'neu': 0.793, 'pos': 0.207, 'compound': 0.9879}
    # {'neg': 0.009, 'neu': 0.81, 'pos': 0.181, 'compound': 0.9859}

    # Append two lists of text and sentiment scores
    data.append(unique_string)
    scores.append(sia.polarity_scores(unique_string))

# Create the DataFrame
df = pd.DataFrame(scores)

# Insert the list of dates into the first column and call it Date
df.insert(0, 'Date', dates)
#print(df.head(10))

# Define the ticker symbol and create a Ticker object
ticker_symbol = 'AAPL'
ticker = yf.Ticker(ticker_symbol)

# Fetch historical market data and make dataframe df1
historical_data = ticker.history(period="1y")
df1 = pd.DataFrame(historical_data)

#print(df1.head(10))
#print(df1.info())

# Make a new datetime index for df to match df1 columns for a plot
df['time'] = '00:00:00-5:00'
df['Date'] = df['Date'].astype(str)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S%z')
df = df.set_index(pd.DatetimeIndex(df['Datetime']))
#print(df.head(10))

#plot with two y-axes
col1 = 'steelblue'
col2 = 'red'
fig,ax = plt.subplots()
ax.plot(df1.index, df1['Close'], color=col1)
ax.set_xlabel('Day', fontsize=14)
ax.set_ylabel('Closing Price', color=col1, fontsize=16)
ax2 = ax.twinx()
ax2.plot(df1.index, df1['compound'], color=col2)
ax2.set_ylabel('Sentiment Index', color=col2, fontsize=16)
plt.show()
    

