#Robert Kaszubski
#rkaszubs@depaul.edu

#Ashish Gare
#agare@depaul.edu


#APP IMPORTS
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.properties import StringProperty, ListProperty

#SCRAPER IMPORTS
import twint



#ANALYSIS IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import html
import re
import preprocessor as pre
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
from collections import Counter
import itertools
import collections
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud
import random
import seaborn as sns
from nrclex import NRCLex

#SET MINIMUM WINDOW SIZE (DESKTOP APP)
Window.size = (1280, 720)
Window.minimum_width, Window.minimum_height = Window.size


#Popup to ask user to wait during scraping
class MyPopup(Popup):
    def __init__(self, **kwargs):
        super(MyPopup, self).__init__(**kwargs)

    def dismiss_popup(self):
        self.dismiss()
    def open_popup(self):
        self.open()

#Home Screen
class Home(Screen):
    pass

#Use Saved/Existing Data Screen
class Existing(Screen):
    def __init__(self, **kwargs):
        super(Existing, self).__init__(**kwargs)
        #initialize currentfile store
        self.currentfile = ""

    def on_pre_enter(self):
        self.ids.filechooser._update_files()

    #Function to set currentfile to user chosen file
    def set_file(self,select):
        #print(select[0])
        self.currentfile = select[0]
        print(self.currentfile)
    pass

#Exploratory Analysis Screen
class Explore(Screen):
    def __init__(self, **kwargs):
        super(Explore, self).__init__(**kwargs)
        #initialize data store
        self.data = pd.DataFrame()
        self.ps = nltk.PorterStemmer()
        self.wn = nltk.WordNetLemmatizer()
        self.final_data = pd.DataFrame()
        self.currlike = 0
        self.currretweet = 0
        self.currreptweet = 0
        self.likes = []
        self.retweets = []
        self.reptweets = []


    def on_pre_enter(self):
        self.read_file()
        self.langs()
        self.currlike = 0
        self.currretweet = 0
        self.currreptweet = 0
        self.liked_tweets()
        self.liked_inc(0)
        self.re_tweets()
        self.retweet_inc(0)
        self.rep_tweets()
        self.rep_inc(0)
        self.nlp_run()

        #read csv to pandas df
    def read_file(self):
        #print(self.manager.get_screen('existing').currentfile)
        cf = self.manager.get_screen('existing').currentfile
        self.data = pd.read_csv(cf, lineterminator='\n')
        #print(self.data.iloc[0])

        #incredment liked tweets
    def liked_inc(self, inc):
        curr = self.currlike
        if curr + inc <= 0:
            self.currlike = 0
        elif curr + inc >= 9:
            self.currlike = 9
        else:
            self.currlike = curr + inc
        self.ids.top_like.text = self.likes[self.currlike]

        #retrieve 10 most liked tweets
    def liked_tweets(self):
        likedtweets = []
        tweet_likes_count = self.data.sort_values(by='likes_count', ascending=False)
        tweet_likes_count = tweet_likes_count.reset_index(drop=True)
        for i in range(10):
            out_string = "Username: {} \nLikes: {}\nTweet: {}".format(
                tweet_likes_count['username'][i],
                tweet_likes_count['likes_count'][i],
                tweet_likes_count['tweet'][i]
            )
            likedtweets.append(out_string)
        self.likes = likedtweets

        #increment retweets
    def retweet_inc(self, inc):
        curr = self.currretweet
        if curr + inc <= 0:
            self.currretweet = 0
        elif curr + inc >= 9:
            self.currretweet = 9
        else:
            self.currretweet = curr + inc
        self.ids.top_retweet.text = self.retweets[self.currretweet]

        #retrieve 10 most retweets tweets
    def re_tweets(self):
        retweets = []
        tweet_likes_count = self.data.sort_values(by='retweets_count', ascending=False)
        tweet_likes_count = tweet_likes_count.reset_index(drop=True)
        for i in range(10):
            out_string = "Username: {} \nRetweets: {}\nTweet: {}".format(
                tweet_likes_count['username'][i],
                tweet_likes_count['retweets_count'][i],
                tweet_likes_count['tweet'][i]
            )
            retweets.append(out_string)
        self.retweets = retweets

        #increment replies
    def rep_inc(self, inc):
        curr = self.currreptweet
        if curr + inc <= 0:
            self.currreptweet = 0
        elif curr + inc >= 9:
            self.currreptweet = 9
        else:
            self.currreptweet = curr + inc
        self.ids.top_rep.text = self.reptweets[self.currreptweet]

        #retrieve 10 most replied tweets
    def rep_tweets(self):
        reptweets = []
        tweet_likes_count = self.data.sort_values(by='replies_count', ascending=False)
        tweet_likes_count = tweet_likes_count.reset_index(drop=True)
        for i in range(10):
            out_string = "Username: {} \nReplies: {}\nTweet: {}".format(
                tweet_likes_count['username'][i],
                tweet_likes_count['replies_count'][i],
                tweet_likes_count['tweet'][i]
            )
            reptweets.append(out_string)
        self.reptweets = reptweets

    def autopct(self,pct): # only show the label when it's > 10%
        return ('%.2f' % pct) if pct > 10 else ''

        #get language data and generate bar plot
    def langs(self):
        plt.clf()
        plt.close()
        #get language data
        lang = self.data['language']
        lang = lang.value_counts()

        #send total count to kv file
        self.ids.langcnt.text = "Total Number of Languages: {}".format(lang.count())
        #lang = lang[:8]
        #check if there is more than 5 languages
        try:
            lang = lang[:5]
        except:
            pass

        #create plot
        values = list(lang.values)
        keys = list(lang.keys())
        plt.clf()
        plt.close()
        plt.figure( figsize=(20,10))
        sns.set( style = "dark")
        sns.set(font_scale = 3)
        ax = sns.barplot(y=keys, x=values, orient = 'h')
        ax.set_xlabel("Frequency")
        #save temporarily and display plot
        plt.savefig('temp/lang.png', bbox_inches='tight')
        self.ids.langplot.reload()



        #nlp helpers
    def tokenization(self, text):
        text = re.split('\W+', text)
        return text

    def stemming(self, text):
        text = [self.ps.stem(word) for word in text]
        return text

    def lemmatizer(self,text):
        text = [self.wn.lemmatize(word) for word in text]
        return text

    def get_tweet_sentiment(self,text):
        return text

        #main nlp function to process data
    def nlp_run(self):
        data_new = self.data

        #remove dups - unneeded as scrapper already does this
        #data_new = data_new.drop_duplicates('tweet', keep='first')

        #remove non english
        #data_final = data_new[data_new['language'] == 'en' ]
        data_final = data_new

        reg = re.compile("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z t])|(w+://S+)") #special characters
        data_final['tweet'] = data_final['tweet'].replace(reg,'',regex=True)
        data_final["tweet"] = data_final["tweet"].str.lower() #lowercase

        #stopwords
        stop_words = stopwords.words('english') #preparing stopwords
        stop_words.remove('not')
        data_final['tweet'] = data_final['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

        reg2 = re.compile('((www.[^s]+)|(https?://[^s]+))') #remove HTML
        data_final['tweet'] = data_final['tweet'].replace(reg2,'',regex=True)

        reg3 = re.compile('[0-9]+') #remove numbers!
        data_final['tweet'] = data_final['tweet'].replace(reg3,'',regex=True)

        #tokenization
        data_final['Tweet_tokenized'] = data_final['tweet'].apply(lambda x: self.tokenization(x.lower()))
        data_final.head()

        #stemming
        data_final['Tweet_stemmed'] = data_final['Tweet_tokenized'].apply(lambda x: self.stemming(x))
        data_final.head()

        #Lemmatization
        data_final['Tweet_lemmatized'] = data_final['Tweet_stemmed'].apply(lambda x: self.lemmatizer(x))
        data_final.head()

        data_final['date'] = data_final['created_at'].str[:10]
        data_final[["year", "month", "day"]] = data_final["date"].str.split("-", expand = True)



        data_final['tweet_content'] = data_final['Tweet_tokenized'].apply(lambda x: self.get_tweet_sentiment(' '.join(x)))
        data_final['emotions']=data_final['tweet_content'].apply(lambda x: NRCLex(x).affect_frequencies)
        self.final_data = data_final

        stem = data_final[data_final['language'] == 'en']['Tweet_stemmed']
        words_no_stopwords = list(itertools.chain(*stem))
        self.ids.wrdcnt.text = "Total Number of Words Without Stop Words: {}".format(len(words_no_stopwords))
        counts_no_stopwords = collections.Counter(words_no_stopwords)
        out_words = counts_no_stopwords.most_common(5)
        keys = []
        values = []
        for tup in out_words:
            keys.append(tup[0])
            values.append(tup[1])
        #print(keys,values)
        #Frequency bar chart
        plt.clf()
        plt.close()
        plt.figure( figsize=(20,10))
        sns.set( style = "dark")
        sns.set(font_scale = 3)
        ax = sns.barplot(y=keys, x=values, orient = 'h')
        ax.set_xlabel("Frequency")
        plt.savefig('temp/words.png', bbox_inches='tight')
        self.ids.wordplot.reload()

        #WordCloud
        wc = self.final_data[self.final_data['language'] == 'en']
        text = " ".join(i for i in wc.tweet_content)
        stop_words = set(stopwords.words("english"))
        wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(text)
        plt.clf()
        plt.close()
        plt.figure( figsize=(20,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('temp/cloud.png', bbox_inches='tight')
        self.ids.cloud.reload()

#sentiment Analysis Screen
class Sentiment(Screen):
    def __init__(self, **kwargs):
        super(Sentiment, self).__init__(**kwargs)
        #initialize data store
        self.final_data = pd.DataFrame()
        self.split = 10

    def on_pre_enter(self):
        self.get_data()
        self.split = 10
        self.ids.split.text = str(self.split)
        self.sentiment_plots()

        #fetch data from previous screen
    def get_data(self):
        self.final_data = self.manager.get_screen('explore').final_data
        self.final_data = self.final_data[self.final_data['language'] == 'en']
        #print(self.final_data.iloc[0])

        #interactive split control
    def update_split(self, inc):
        curr = self.split
        if curr + inc <= 0:
            self.split = 1
        elif curr + inc > 20:
            self.split = 20
        else:
            self.split = curr + inc
        self.ids.split.text = str(self.split)
        self.sentiment_plots()

        #helper function to convert to datetime object
    def to_date(self,datestring):
        datestring = datestring[:19]
        formater = "%Y-%m-%d %H:%M:%S"
        timestamp = datetime.datetime.strptime(datestring, formater)
        return timestamp

        #function to create 4 sentiment plots based on user input of splits
    def sentiment_plots(self):
        N = self.split
        df = pd.concat([self.final_data.drop(['emotions'], axis=1), self.final_data['emotions'].apply(pd.Series)], axis =1)
        df = df[::-1]
        df['created_at'] = self.final_data['created_at'].apply(lambda x: self.to_date(x))

        #get first and last timestamp in dataset
        timestamp= df['created_at'].iloc[0]
        timestamp2= df['created_at'].iloc[-1]

        #split into N date ranges:
        diff = (timestamp2 - timestamp) //N
        dates = []
        for idx in range(0, N):
            # computing new dates
            dates.append((timestamp + idx * diff))
        dates.append(timestamp2)

        #get sentiment results:
        results_sum = []
        results_avg = []
        results_sum_pn = []
        results_avg_pn = []
        for date in range(0,len(dates)-1):
            df_dates = df.loc[(df['created_at'] >= dates[date]) & (df['created_at'] <= dates[date+1])]
            df_dates_emo = df_dates[['fear','anger','trust','surprise','sadness','disgust','joy','anticipation']]
            df_dates_emo = df_dates_emo.fillna(0)
            results_sum.append((list(df_dates_emo.sum().values)))
            results_avg.append((list(df_dates_emo.mean().values)))
            df_dates_pn = df_dates[['positive','negative']]
            df_dates_pn = df_dates_pn.fillna(0)
            results_sum_pn.append((list(df_dates_pn.sum().values)))
            results_avg_pn.append((list(df_dates_pn.mean().values)))

        #convert dates for plotting:
        plot_dates = []
        for date in dates[1:]:
            item = date.strftime("%Y-%m-%d %H:%M:%S")
            out = item[:10]+"\n"+item[10:]
            plot_dates.append(out)
        #plot_dates

        #emotion sum plot:
        out_df = pd.DataFrame(results_sum, columns = ['fear','anger','trust','surprise','sadness','disgust','joy','anticipation'])
        plt.clf()
        plt.close()
        plt.figure(figsize=(20,8))
        sns.set(font_scale = 3)
        sns.set_style("dark")
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 8)
        plt.plot(out_df,  marker='o',linestyle = '--', linewidth=4, markersize=10)
        #plt.legend(['Fear', 'Anger','Trust', 'Surprise', 'Sadness', 'Digust', 'Joy', 'Anticipation'], title = "Legend", bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xticks(range(N),plot_dates,rotation = 90)
        plt.ylabel("Sum Score")
        #ax.set_xticklabels(plot_dates)
        plt.savefig('temp/sent1.png', bbox_inches='tight')
        plt.clf()
        plt.close()
        self.ids.sent1.reload()

        #emotion avg plot:
        out_df = pd.DataFrame(results_avg, columns = ['fear','anger','trust','surprise','sadness','disgust','joy','anticipation'])
        plt.clf()
        plt.close()
        plt.figure(figsize=(20,8))
        sns.set(font_scale = 3)
        sns.set_style("dark")
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 8)
        plt.plot(out_df,  marker='s',linestyle = '--', linewidth=4, markersize=10)
        #plt.legend(['Fear', 'Anger','Trust', 'Surprise', 'Sadness', 'Digust', 'Joy', 'Anticipation'],  bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xticks(range(N),plot_dates,rotation = 90)
        plt.ylabel("Avg Score")
        #ax.set_xticklabels(plot_dates)
        plt.savefig('temp/sent2.png', bbox_inches='tight')
        plt.clf()
        plt.close()
        self.ids.sent2.reload()

        #sent sum plot:
        out_df = pd.DataFrame(results_sum_pn, columns = ['positive','negative'])
        plt.clf()
        plt.close()
        plt.figure(figsize=(20,8))
        sns.set(font_scale = 3)
        sns.set_style("dark")
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 8)
        plt.plot(out_df,  marker='o', linewidth=4, markersize=10)
        #plt.legend(['Positive', 'Negative'], title = "Legend", bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xticks(range(N),plot_dates,rotation = 90)
        plt.ylabel("Sum Score")
        #ax.set_xticklabels(plot_dates)
        plt.savefig('temp/sent3.png', bbox_inches='tight')
        plt.clf()
        plt.close()
        self.ids.sent3.reload()

        #sent avg plot:
        out_df = pd.DataFrame(results_avg_pn, columns = ['positive','negative'])
        plt.clf()
        plt.close()
        plt.figure(figsize=(20,8))
        sns.set(font_scale = 3)
        sns.set_style("dark")
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 8)
        plt.plot(out_df,  marker='o', linewidth=4, markersize=10)
        #plt.legend(['Positive', 'Negative'], bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xticks(range(N),plot_dates,rotation = 90)
        plt.ylabel("Avg Score")
        #ax.set_xticklabels(plot_dates)
        plt.savefig('temp/sent4.png', bbox_inches='tight')
        plt.clf()
        plt.close()
        self.ids.sent4.reload()

#Tweet Anaylsis Screen
class Tweet(Screen):
    def __init__(self, **kwargs):
        super(Tweet, self).__init__(**kwargs)
        #initialize data store
        self.final_data = pd.DataFrame()
        self.data = pd.DataFrame()
        self.current = 0
        self.sim_emotion = np.array(0)
        self.sim_content = np.array(0)

    def on_pre_enter(self):
        self.get_data()
        self.current = 0
        self.getsims()
        self.show_selected()
        self.tfidf(0)

        #fetch data from explore screen
    def get_data(self):
        self.final_data = self.manager.get_screen('explore').final_data
        self.data = self.manager.get_screen('explore').data
        #print(self.final_data.iloc[0])
        #print(self.data.iloc[0])


        #function called to select random tweet
    def random_tweet(self):
        lentweets = self.data.shape[0]
        self.current = random.randint(0,lentweets-1)
        self.show_selected()

        #function called to read user input and select tweet
    def selected_tweet(self):
        lentweets = self.data.shape[0]
        try:
            id = int(self.ids.tweetid.text)
            if id >= 0 and id < lentweets:
                self.current = id
                self.show_selected()
        except:
            pass

        #function to navigate tweets
    def increment(self, inc):
        lentweets = self.data.shape[0]
        curr = self.current
        if curr + inc <= 0:
            self.current = 0
        elif curr + inc >= lentweets-1:
            self.current = lentweets-1
        else:
            self.current = curr + inc
        #print(self.data.shape[0])
        #print(self.final_data.shape[0])
        self.show_selected()

        #function to display selected tweet
    def show_selected(self):
        tweets = self.data[['tweet','username','created_at','likes_count','replies_count','retweets_count']]
        id = self.current
        tweet_string = "Tweet ID: {} \nUsername: {} \nDate: {}\nLikes: {}, Replies: {}, Retweets: {} \n\nTweet: {}".format(
            id,
            tweets['username'][id],
            tweets['created_at'][id][:10],
            str(tweets['likes_count'][id]),str(tweets['replies_count'][id]),str(tweets['retweets_count'][id]),
            tweets['tweet'][id]
            )
        self.ids.seltweet.text = tweet_string
        self.tfidf(id)
        self.nrc()
        self.sim_emo()
        #self.ids.boxemocontent._trigger_layout()
        #self.ids.scrollemocontent.update_from_scroll()

        #helper function for tfidf
    def normalize_document(self,doc):
        wpt = nltk.WordPunctTokenizer()
        stop_words = nltk.corpus.stopwords.words('english')
        ps = nltk.stem.PorterStemmer()
        # convert to lower case, and remove special characters and white space
        doc = re.sub(r'[^a-zA-Z0-9_\s]', '', doc, re.I)
        doc = doc.lower()
        doc = doc.strip()
        # tokenize the document
        tokens = wpt.tokenize(doc)
        # remove stopwords
        filtered_tokens = [token for token in tokens if (token not in stop_words and token not in ["."])]
        # put the filtered document back together
        doc = ' '.join([ps.stem(token) for token in filtered_tokens])
        return doc

        #function for content based similarity
    def tfidf(self, item):
        tweets = self.data[['tweet','username']]
        sims = self.sim_content[item,:]
        idx = np.argsort(sims)
        idx = idx[::-1]
        ### Need to remove the item itself since it has the highest similarity to itself
        idx = np.array([i for i in idx if i != item])
        neigh_idx = idx[:5]
        neigh_sims = sims[neigh_idx]
        out_string = ''
        for x in range(0,5):
            tweet_string = "Tweet ID: {} - Similarity Value: {} \nUsername: {} \nTweet: {}\n\n".format(
                neigh_idx[x],
                neigh_sims[x],
                tweets['username'][neigh_idx[x]],
                tweets['tweet'][neigh_idx[x]]
                )
            out_string += tweet_string
        self.ids.contenttweets.text = out_string

        #function for emotion plot
    def nrc(self):
        emotions = self.final_data['emotions'].iloc[self.current]
        #print(self.data.shape, self.final_data.shape)
        #nrc = NRCLex(tweet).affect_frequencies
        plt.clf()
        plt.close()
        plt.figure( figsize=(20,10))
        values = list(emotions.values())
        keys = list(emotions.keys())
        #add anticipation to the end if it returns NAN
        if keys[-1] != "anticipation":
            keys.append("anticipation")
            values.append(0)
        #print(values, keys)
        #remove anticip
        del values[2]
        del keys[2]
        #print(values, keys)
        sns.set( style = "dark")
        sns.set(font_scale = 3)
        ax = sns.barplot(y=keys, x=values, orient = 'h')
        ax.set_xlabel("Emotional Affect", fontsize=30)
        plt.savefig('temp/nrc.png', bbox_inches='tight')
        self.ids.nrcplot.reload()

        #function for emo/sentiment and content based tfidf similarity matricies
    def getsims(self):
        emotions = np.array(self.final_data['emotions'].apply(pd.Series).fillna(0))
        sim_mat = cosine_similarity(emotions)
        self.sim_emotion = sim_mat
        vectorizer = TfidfVectorizer(preprocessor=self.normalize_document, norm=None, max_df=0.8, min_df=3)
        vectorizer.fit(self.data['tweet'])
        tweet_mat = vectorizer.transform(self.data['tweet'])
        simMatrix = cosine_similarity(tweet_mat)
        self.sim_content = simMatrix

        #function to display most similar tweets by emotion
    def sim_emo(self):
        item = self.current
        sims = self.sim_emotion[item,:]
        idx = np.argsort(sims)
        idx = idx[::-1]
        idx = np.array([i for i in idx if i != item])
        neigh_idx = idx[:5]
        neigh_sims = sims[neigh_idx]
        out_string = ''
        tweets = self.data[['tweet','username']]
        for x in range(0,5):
            tweet_string = "Tweet ID: {} - Similarity Value: {} \nUsername: {} \nTweet: {}\n\n".format(
                neigh_idx[x],
                neigh_sims[x],
                tweets['username'][neigh_idx[x]],
                tweets['tweet'][neigh_idx[x]]
                )
            out_string += tweet_string
        self.ids.emotweets.text = out_string



#Scraping Screen
class Scraper(Screen):
    def __init__(self, **kwargs):
        super(Scraper, self).__init__(**kwargs)
        #initialize popup object
        self.pop = MyPopup()

        #close popup
    def close(self):
        self.pop.dismiss_popup()

        #show popup
    def open(self):
        self.pop.open_popup()

        #convert user keyword input into search query
    def get_search_str(self, keywords):
        keywords = list(keywords.split(','))
        search_str = ''
        for keyword in keywords[:-1]:
            word = "'" + keyword.strip(" ") + "'"
            search_str += word + ' OR '
        search_str += ("'" + keywords[-1].strip(" ") + "'")
        return search_str

        #run twint scraper
    def keyword_scrape(self):

        keywords = self.get_search_str(self.ids.keywords.text)

        limit = 100
        try:
            limit = int(self.ids.limit.text)
            if limit < 20:
                limit = 20
        except:
            pass

        since = ""
        until = ""
        try:
            #ensures proper format
            formater = "%Y-%m-%d"
            since = self.ids.fromdate.text
            until = self.ids.untildate.text
            try:
                datetime_since = datetime.datetime.strptime(since, formater)
                datetime_until = datetime.datetime.strptime(until, formater)
            except:
                since = ""
                until = ""
        except:
            pass

        like_cnt = 0
        try:
            like_cnt = int(self.ids.like_lim.text)
        except:
            pass

        replies_cnt = 0
        try:
            replies_cnt = int(self.ids.replies_lim.text)
        except:
            pass

        retweets_cnt = 0
        try:
            retweets_cnt = int(self.ids.retweets_lim.text)
        except:
            pass

        filename = self.ids.name.text
        print(keywords)
        print(limit)
        print(since)
        print(until)
        print(filename)
        #self.keyword_scrape(limit, keywords, since, until, filename)
        currentfile = "data//{}.csv".format(filename)
        print(currentfile)
        ## Create Search Term using all Keywords
        #search_str = get_search_str(keywords)

        ## Configure Scraper Tweets
        c = twint.Config()
        c.Hide_output = True
        if since and until:
            c.Since = since
            c.Until = until
        c.Search = keywords
        c.Min_likes = like_cnt
        c.Min_replies = replies_cnt
        c.Min_retweets = retweets_cnt
        c.Limit = limit
        c.Retweets = False
        c.Count = True

        #store tweets
        c.Custom["tweet"] = ["id","created_at","username","likes_count","replies_count","retweets_count","language","tweet","hashtags"]
        c.Store_csv = True
        c.Output = "data//{}.csv".format(filename)
        twint.run.Search(c)





class WindowManager(ScreenManager):
    pass

#load kv file
kv = Builder.load_file('tsa.kv')

#main class
class TwitterScraperAnalyzer(App):
    def build(self):
        self.icon = "assets/logo.png"
        return kv



if __name__ == '__main__':
    TwitterScraperAnalyzer().run()
