import re
from datetime import datetime, timedelta
import requests

# NLP and map sections
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import pandas as pd
import datefinder
import yfinance as yf
#import investpy as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet

# User inputs (up to 5 years)
years_in = int(input("Enter the number of years: "))
#years_in = 2

# Get current year through datetime module
current_year = datetime.now().year

# Sample list of words
q_words = ['first','second','third','fourth']
m_words = ["/02/", "/05/", "/08/", "/10/"]
y_words = [str(current_year), str(current_year - 1), str(current_year - 2)]

# Convert list to a regex pattern (case-insensitive search)
q_pattern = r'\b(?:' + '|'.join(map(re.escape, q_words)) + r')\b'
m_pattern = r'\b(?:' + '|'.join(map(re.escape, m_words)) + r')\b'
y_pattern = r'\b(?:' + '|'.join(map(re.escape, y_words)) + r')\b'

# Sample URL (AAPL)
text = "https://www.apple.com/newsroom/2024/02/apple-reports-first-quarter-results/"

# Other urls to try...others from top 5 S&P500 companies: NVDA, MSFT, AMZN, META.
#NVDA's 4th quarter is different than other urls: https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Announces-Financial-Results-for-Fourth-Quarter-and-Fiscal-2024/
#text = "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2025"
#Microsoft's urls have a Q1, Q2, Q3, and Q4.
#text = "https://www.microsoft.com/en-us/Investor/earnings/FY-2025-Q1/press-release-webcast"
#Amazon's urls yield 'status code: 403' meaning access to the requested resource is forbidden.
#text = "https://ir.aboutamazon.com/news-release/news-release-details/2024/Amazon.com-Announces-Third-Quarter-Results/"
#Facebook's urls also yield 'status code: 403' meaning access to the requested resource is forbidden.
#text = "https://investor.fb.com/investor-news/press-release-details/2024/Meta-Reports-Third-Quarter-2024-Results/default.aspx"

# Index of the first occurance of 'apple'
apple = text.find('apple')

# Find all the matches using findall
q_matches = re.findall(q_pattern, text, re.IGNORECASE)
m_matches = re.findall(m_pattern, text, re.IGNORECASE)
y_matches = re.findall(y_pattern, text, re.IGNORECASE)

# Dictionaries for quarter and month and year
if q_matches[0][0].isupper():
    qtr_dict = {
        "First": ['First','Second','Third','Fourth'],
        "Second": ['Second','Third','Fourth','First'],
        "Third": ['Third', 'Fourth','First','Second'],
        "Fourth": ['Fourth','First','Second','Third']
    }
elif q_matches[0][0].islower():
        qtr_dict = {
        "first": ['first','second','third','fourth'],
        "second": ['second','third','fourth','first'],
        "third": ['third', 'fourth','first','second'],
        "fourth": ['fourth','first','second','third']
    }
        
mon_dict = {
    "/02/": ["/02/", "/05/", "/08/", "/10/"],
    "/05/": ["/05/", "/08/", "/10/", "/02/"],
    "/08/": ["/08/", "/10/", "/02/", "/05/"],
    "/10/": ["/10/", "/02/", "/05/", "/08/"],
}

# Set up year dictionary for 1, 2, 5, and 10 years according to historical stock prices from yfinance's ticker.history(period="")
y_dict = {
    1: [y_matches[0]],
    2: [y_matches[0], str(int(y_matches[0]) - 1)],
    5: [y_matches[0], str(int(y_matches[0]) - 1), str(int(y_matches[0]) - 2), str(int(y_matches[0]) - 3), str(int(y_matches[0]) - 4)],
    10: [y_matches[0], str(int(y_matches[0]) - 1), str(int(y_matches[0]) - 2), str(int(y_matches[0]) - 3), str(int(y_matches[0]) - 4), str(int(y_matches[0]) - 5), str(int(y_matches[0]) - 6), str(int(y_matches[0]) - 7), str(int(y_matches[0]) - 8),str(int(y_matches[0]) - 9)],
}

url_list = []
text_m =[]
for y in y_dict[years_in]:
      text_y = text.replace(y_matches[0], y)
      for q in qtr_dict[q_matches[0]]: # if condition for quarters
            text_q = text_y.replace(q_matches[0], q)
            qindex = qtr_dict[q_matches[0]].index(q)
            if m_matches != []:
                  text_m = text_q.replace(m_matches[0], mon_dict[m_matches[0]][qindex])
                  try:
                        response = requests.get(text_m)
                        if response.status_code == 200:
                              #print(f"URL is good: {text_m}")
                              url_list.append(text_m)
                        else:
                              print(f"URL is not good (status code: {response.status_code}): {text_m}")
                              plus_1 = '/'+str(int(mon_dict[m_matches[0]][qindex].strip('/'))+1).zfill(2) + '/'
                              text_m = text_m.replace(mon_dict[m_matches[0]][qindex], plus_1)
                              response = requests.get(text_m)
                              if response.status_code == 200:
                                #print(f"URL is good: {text_m}")
                                url_list.append(text_m)
                              else:
                                    print(f"URL is not good (status code: {response.status_code}): {text_m}")
                  except requests.exceptions.RequestException as e:
                        print(f"Error with URL {text_m}: {e}")
            else:
                  response = requests.get(text_q)
                  if response.status_code == 200:
                        #print(f'URL is good: {text_q}')
                        url_list.append(text_q)
                  else:
                    print(f"URL is not good (status code: {response.status_code}): {text_m}")

print('You have ' + str(len(url_list)) + ' quarterly reports.')

# NLP Section
data = []
siascores = []
vaderscores = []
tblobscores = []
tblobscores2 = []
afinnscores = []
dates = []

for url in url_list:
    text = ''
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    date_tag = soup.find('p')
    # view page source for details on NVDA
    if apple != -1:
        text_all = soup.find_all(attrs={'class': 'pagebody-copy'})
        for element in text_all:
              text_0 = element.get_text(' ', strip=True)
              text = text + text_0
    else:
      body = soup.body
      if body is not None:
        p_tags = body.find_all('p', limit=6)
      else:
        p_tags = soup.find_all('p', limit=6)    
      for p in p_tags:
           text_0 = p.text
           #print(text_0)
           text = text + text_0

    sent = sent_tokenize(text)
    words = [word_tokenize(t) for t in sent]
    list_words = sum(words, [])
    lower_words = [w.lower() for w in list_words]
    remove_words = [w for w in lower_words if w not in stopwords.words('english')]
    punctuation_words = [w for w in remove_words if w.isalnum()]

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

    # Use datefinder to find first match in text. Then break becuase date always at beignning of quarterly release
    if date_tag:
        dt = date_tag.get_text(" ", strip=True)
        matches = list(datefinder.find_dates(unique_string))
    # if matches:
    #      dates.append(matches[0])
    # else:
    #      dates.append(None)
    for match in matches:
        break
    dates.append(match)

    # match = next(matches, None)
    # if match is not None:
    #     dates.append(match)


    # Use sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    # VADER (Valence Aware Dictionary and sEntiment Reasoner) is a sentiment analysis tool optimized for social media and financial text.
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    vsent = vader.polarity_scores(unique_string) # ranges from -1 to 1, where -1 is negative, 0 is neutral, and 1 is positive.

    from textblob import TextBlob
    blob = TextBlob(unique_string)
    bsent = blob.sentiment

    from afinn import Afinn
    afinn = Afinn()
    asent = afinn.score(unique_string) # This gives a score which is the sum of individual word scores. Rangign from negative(unhappy) to positive(happy).

    # Create dataframes from two sentiment analyzers, concat, add new columns for other two sentiment intensity analyzers.
    siascores.append(sia.polarity_scores(unique_string))
    vaderscores.append(vsent)
    tblobscores.append(bsent.polarity) # ranges from -1 to 1, where -1 is negative, 0 is neutral, and 1 is positive. Tells you how positive or negative the report is.
    tblobscores2.append(bsent.subjectivity) # ranges from 0 to 1, where 0 is objective and 1 is subjective. Teels how much of the report is opinion vs fact
    afinnscores.append(asent)

df1 = pd.DataFrame(siascores)
df1.columns = ['sia_neg','sia_neu','sia_pos','sia_compound']

dfvader = pd.DataFrame(vaderscores)
dfvader.columns = ['vader_neg','vader_neu','vader_pos','vader_compound']
df1 = pd.concat([df1, dfvader], axis=1)

df1['Afinn'] = afinnscores
df1['tblob_polarity'] = tblobscores
df1['tblob_subjectivity'] = tblobscores2

# Append two list of text and sentiment scores
data.append(unique_string)

# Insert the list of dates into the first column (0) and call it Date
print(len(dates), dates)
print(len(df1))
df1.insert(0, 'Date', dates)

# sort the data by Date and print
df1 = df1.sort_values(by='Date')
print(df1.head(10))


# The problem we are facing right now is with yfinance, which runs out of request constantly
# ---------------------------------------------------------------------------------------------------------------#
# To overcome this, we are going to use investpy and define a ticker function itself which will replicate this 

# This is the last thing i made for the investpy, it was good but led to a ConnectionError(), I don't know why
# class Ticker:
#     def __init__(self, stock: str, country='united states'):
#         self.stock = stock
#         self.country = country

#     def history(self, period: str = "1y", interval: str = "Daily") -> pd.DataFrame:
#         # Convert period like '5y', '6mo', '30d' into start_date and end_date
#         end_date = datetime.today()
        
#         if period.endswith("y"):  # years
#             years = int(period[:-1])
#             start_date = end_date - timedelta(days=years * 365)
#         elif period.endswith("mo"):  # months
#             months = int(period[:-2])
#             start_date = end_date - timedelta(days=months * 30)
#         elif period.endswith("d"):  # days
#             days = int(period[:-1])
#             start_date = end_date - timedelta(days=days)
#         else:
#             raise ValueError("Unsupported period format. Use like '5y', '6mo', '30d'.")

#         # Format for investpy
#         from_date = start_date.strftime("%d/%m/%Y")
#         to_date = end_date.strftime("%d/%m/%Y")

#         df2 = yf.get_stock_historical_data(
#             stock=self.stock,
#             country=self.country,
#             from_date=from_date,
#             to_date=to_date,
#             interval=interval
#         )
#         return df2


# ticker = Ticker('AAPL')
# historical_data = ticker.history(period=str(years_in) + 'y')
# ---------------------------------------------------------------------------# 

# yfinance section
# Define the ticker symbol and create a ticker object
# yfinance code         
ticker_symbol = 'AAPL'
ticker = yf.Ticker(ticker_symbol)
historical_data = ticker.history(period=str(years_in) + 'y')
df2 = pd.DataFrame(historical_data)

# Make a new datetime index for df1 to match df2 columns for a plot
df1['time'] = '00:00:00-5:00'
df1['Date'] = df1['Date'].astype(str)
df1['Datetime'] = pd.to_datetime(df1['Date'] + ' ' + df1['time'], format='%Y-%m-%d %H:%M:%S%z')
df1 = df1.set_index(pd.DatetimeIndex(df1['Datetime']))


# Create subplots with 2 rows and 1 column
fig, axes = plt.subplots(2, 1, figsize=(8,8))

# First plot with two axes
ax1 = axes[0]
ax1.plot(df2.index, df2['Close'], label="Closing price", color="green")
ax1.set_ylabel("Closing price", color="green")
ax1.tick_params(axis='y', labelcolor="green")
ax1.set_title("SIA and VADER sentiment scores with AAPL closing price")

# Create second y-axis for the first plot
ax2 = ax1.twinx()
ax2.plot(df1.index, df1['sia_compound'], label="SIA Sentiment Index", color="red")
ax2.set_ylabel("SIA Sentiment Index", color="red")
ax2.tick_params(axis='y', labelcolor="red")

# Add legends to avoid overlap
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Second plot with two y-axes
ax3 = axes[1]
ax3.plot(df2.index, df2['Close'], label="Closing price", color="green")
ax3.set_ylabel("Closing price", color="green")
ax3.tick_params(axis='y', labelcolor="green")
ax3.set_xlabel("Date")

# Create second y-axis for the second plot
ax4 = ax3.twinx()
ax4.plot(df1.index, df1['vader_compound'], label="VADER Sentiment Index", color="purple")
ax4.set_ylabel("VADER Sentiment Index", color="purple")
ax4.tick_params(axis='y', labelcolor="purple")

# Add legends to avoid overlap
ax3.legend(loc="upper left")
ax4.legend(loc="upper right")

# Adjust layout
plt.tight_layout()
plt.show()

# Ask the user if they want to export to Excel
user_input = input("Do you want to export to Excel? (Y/N): ").strip().lower()

# Make a column of strings for dates in df2. For export to Excel
df2['date_strings'] = df2.index.strftime('%Y-%m-%d')
df1['date_strings'] = df1.index.strftime('%Y-%m-%d')

# df1 and df2 share the same datetime index, pd.concat can combine them directly. Then reset the datetime index to remove the timezone and export to Excel
merged = pd.concat([df1, df2], axis=1)
merged['datestrings'] = df1['date_strings'].combine_first(df2['date_strings'])
merged = merged.drop(['Date','Datetime','date_strings','time'], axis=1)
merged.insert(0, 'Date',merged.pop('datestrings'))
print(merged.info())

if user_input == 'y':
     print("Exporting to Excel...")
     merged.to_excel('output.xlsx', index=False)
elif user_input == 'n':
    print("Skipping Excel export. ")
else:
     print("Invalid input. Please enter 'y' or 'n'.")



    

     
                
            


                    

                                    
