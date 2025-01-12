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
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

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
    bearer = "AAAAAAAAAAAAAAAAAAAAAK7DWgEAAAAAi7d0%2FnUXa55nietE%2BPkTa%2BGABHU%3DZDewMVcDnvjd0A01ZUWSYg9fhZlsiSLKN2XcX5RJ6C1bB1cZGn"
    consumer_key = "a6qFeda7ma8e2kA0drlgpJqNJ"
    consumer_secret = "ixrUzR5mbvj6ZbkvWAvhJid8uXNjRge40b6rWlqdcwMl4e3HTo" 
    access_token = "1468548120808398850-Ka8T3Kg1GhRDTWvMa09Evq713lR2sF"
    access_token_secret = "7U7XyOLrxRMugWvh3okIacE5fiCSaUFPOJBmz5tLzCwJc"
    
    client = tweepy.Client(bearer_token=bearer, consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token, 
                                access_token_secret=access_token_secret)
    
    query = st.text_input("Kueri yang Dicari", placeholder="Masukkan kueri yang akan diextract dari X")
    num_of_tweet = st.number_input("Banyak Tweet yang Ingin Diambil", min_value=0, max_value=100, placholder="Masukkan banyaknya tweet yang ingin dianalisis")
    
    if st.button("Crawl Data"):
        with st.spinner("Crawling..."):
            time.sleep(3.5)
            result = client.search_recent_tweets(query)
            st.success("Done!")
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
            sent_df.to_csv(f"{query}.csv", index=False)
            # Autentikasi
        if st.button("Analys Sentiment of Tweet"):
            with st.spinner("Analys..."):
                # time.sleep(3.5)
                st.success("Done!")
                
                sentiment_df = pd.read_csv(f"{query}.csv")
                
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
                
                dataset = sentiment_df.drop(['data_text'], axis=1, inplace=False)
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

                for text in data_sent:
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
                st.bar_chart(data=sentiment_df, x=sentiment_df['naive_bayes_cl'], color='naive_bayes_cl')
            
    
    
    

    