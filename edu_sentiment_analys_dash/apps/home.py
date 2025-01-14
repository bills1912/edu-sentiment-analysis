import re
import nltk
import time
import tweepy
import random
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from textblob import TextBlob
from translate import Translator
from nltk.tokenize import word_tokenize
from textblob.classifiers import NaiveBayesClassifier
from wordcloud import WordCloud, STOPWORDS
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

nltk.download('punkt_tab')

def preprocess_text(text):
        # create stopword factory and stemmer factory
        stopword = StopWordRemoverFactory().create_stop_word_remover()
        stemmer = StemmerFactory().create_stemmer()
        # convert to lower case
        text = text.lower()
        # remove user handle
        text = re.sub("@[\w]*", "", text)
        # remove http links
        text = re.sub("http\S+", "", text)
        # remove digits and spl characters
        text = re.sub("[^a-zA-Z#0-9]", " ", text)
        # remove additional spaces
        text = re.sub("\s+", " ", text)
        text = stopword.remove(text)
        text = stemmer.stem(text)
        
        return text

def convert_eng(tweet):
    translator = Translator(to_lang='en', from_lang='id')
    translation = translator.translate(tweet)
    return translation

def plot_wordcloud(wordcloud):
    fig = plt.figure(figsize=(5, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig, use_container_width=False)

def app():
    st.write("# Selamat Datang di Dashboard Analisis Sentimen Dinas Pendidikan Kota Gunungsitoli")
    st.markdown(
            """
            Aplikasi ini didasarkan pada adanya inisiatif dari Dinas Pendidikan Kota Gunungsitoli dalam melakukan penganalisaan persepsi
            masyarakat Indonesia tentang pendidikan yang ada di Indonesia itu sendiri. Dashboard ini dibuat sebagai bahan untuk refleksi dan
            evaluasi, apakah pelayanan di bidang pendidikan yang telah dilakukan selama ini sudah sesuai dengan apa yang diinginkan oleh 
            masyarakat luas atau masih terdapat beberapa hal yang 

        """
        )
    bearer = "AAAAAAAAAAAAAAAAAAAAAMt3yAEAAAAAMhT0SZJs4Cwv4F0VNYbq4W2Rvr8%3DwkAUmMOjUEOTfF84fEU2hsseAG23F1bE9o6SjLJdRSzxxt8kPv"
    consumer_key = "uhoJGUqvATvKmDiMSLH9KOKxP"
    consumer_secret = "xDnTmvLwwcSXwAlfxZJx1modVgSkDeAQE8eLJgY0hAofcZsb6I" 
    access_token = "1878638103059124224-EevL32wK37ast89DLRXXSMeXmQpmE1"
    access_token_secret = "DUHFmzG6Kq7B1opRIx3h4eWJqSqgSbeslpQ0kSyka91Qq"
    
    client = tweepy.Client(bearer_token=bearer, consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token, 
                                access_token_secret=access_token_secret)
    
    query = st.text_input("Kueri yang Dicari", placeholder="Masukkan kueri yang akan diextract dari X", help="Kueri yang dimasukkan dapat disesuaikan dengan\
                          kebutuhan postingan yang akan dikumpulkan dan dianalisis")
    num_of_tweet = st.number_input("Banyak Tweet yang Ingin Diambil", min_value=0, max_value=100, placeholder="Masukkan banyaknya tweet yang ingin dianalisis")
    
    if st.button("Crawl Data"):
        with st.spinner("Crawling..."):
            time.sleep(3.5)
            result = client.search_recent_tweets(query, max_results=num_of_tweet)
            sentiment_data_all = result.data
            final_data = []
            sent_df = pd.DataFrame()
            for sent_data  in sentiment_data_all:
                final_data.append(preprocess_text(sent_data.text))
            sent_df['data_text'] = final_data
            data_sent = list(sent_df['data_text'].apply(convert_eng))
            polarization = 0

            status = []
            positive_tot = negative_tot = netral_tot = total = 0

            for text in data_sent:
                analysis = TextBlob(text)
                polarization += analysis.polarity
                
                if analysis.sentiment.polarity > 0.0:
                    positive_tot += 1
                    status.append('Positive')
                elif analysis.sentiment.polarity == 0.0:
                    netral_tot += 1
                    status.append('Netral')
                else:
                    negative_tot += 1
                    status.append('Negative')
                total += 1
            
            sent_df['eng_data'] = data_sent
            sent_df['blob'] = status
            sent_df.to_csv(f"../assets/{query}.csv", index=False)
        st.success("Done!")
            # Autentikasi
    if st.button("Analys Sentiment of Tweet"):
        with st.spinner("Analys..."):
            # time.sleep(3.5)
            
            sentiment_df = pd.read_csv("https://raw.githubusercontent.com/bills1912/edu-sentiment-analysis/refs/heads/main/edu_sentiment_analys_dash/assets/edu_sentiment.csv")
            
            all_words = ' '.join([sent for sent in sentiment_df['data_text']])
            sent_wordcloud = WordCloud(
                width=3000,
                height=2000,
                random_state=3,
                background_color='black',
                colormap='Blues_r',
                collocations=False,
                stopwords=STOPWORDS
            ).generate(all_words)
            
            plot_wordcloud(sent_wordcloud)
            
            dataset = sentiment_df.drop(['data_text', 'naive_bayes_cl'], axis=1, inplace=False)
            dataset = [tuple(x) for x in dataset.to_records(index=False)]
            
            set_positif = []
            set_negatif = []
            set_netral = []

            for n in dataset:
                if(n[1] == 'Positive'):
                    set_positif.append(n)
                elif(n[1] == 'Negative'):
                    set_negatif.append(n)
                else:
                    set_netral.append(n)

            set_positif = random.sample(set_positif, k=int(len(set_positif)/2))
            set_negatif = random.sample(set_negatif, k=int(len(set_negatif)/2))
            set_netral = random.sample(set_netral, k=int(len(set_netral)/2))

            train = set_positif + set_negatif + set_netral

            train_set = []

            for n in train:
                train_set.append(n)
            
            cl = NaiveBayesClassifier(train_set)
            
            polarization = 0

            status_nb = []
            positive_tot = negative_tot = netral_tot = total = 0

            for text in list(sentiment_df['eng_data']):
                analysis = TextBlob(text, classifier=cl)
                polarization += analysis.polarity
                
                if analysis.sentiment.polarity > 0.0:
                    positive_tot += 1
                    status_nb.append('Positive')
                elif analysis.sentiment.polarity == 0.0:
                    netral_tot += 1
                    status_nb.append('Netral')
                else:
                    negative_tot += 1
                    status_nb.append('Negative')
                total += 1
            sentiment_df['naive_bayes_cl'] = status_nb
            fig = plt.figure(figsize=(5, 3))
            st.write(f"#### Analisis Sentimen dari Postingan yang Bersumber dari X dengan kata kunci 'kualitas pendidikan'")
            plt.xlabel('Analisis Sentimen')
            plt.ylabel('Jumlah Postingan')
            sns.countplot(data=sentiment_df, x=sentiment_df['naive_bayes_cl'], hue='naive_bayes_cl')
            st.pyplot(fig, use_container_width=False)
            dict_sentiment = {"Positive":list(sentiment_df[sentiment_df['naive_bayes_cl']=='Positive']['data_text']), 
                              "Negative":list(sentiment_df[sentiment_df['naive_bayes_cl']=='Negative']['data_text']),
                              "Netral":list(sentiment_df[sentiment_df['naive_bayes_cl']=='Netral']['data_text'])}
            for sent_key in dict_sentiment.keys():
                with st.expander(f"{sent_key} Sentiment Tweets"):
                    # hide = """
                    #     <style>
                    #     ul.streamlit-expander {
                    #         overflow: scroll;
                    #         width: 1500px;
                    #     </style>
                    #     """

                    # st.markdown(hide, unsafe_allow_html=True)
                    for sent_tweet in dict_sentiment[sent_key]:
                        st.write(sent_tweet)
                css='''
                <style>
                    [data-testid="stExpander"] div:has(>.streamlit-expanderContent) {
                        overflow: scroll;
                        height: 400px;
                    }
                </style>
                '''

                st.markdown(css, unsafe_allow_html=True)
        st.success("Done!")
            
            
    
    
    

    