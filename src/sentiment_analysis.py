#!/usr/bin/env python

"""
==========================================
Sentiment Analysis on Australian Headlines 
==========================================

This script compares the sentiment analysis between SpacyTextBlob and NLTK's VADER on a dataset of 12 million Australian headlines collected between 2003 and 2020. The script first runs the headlines through the sentiment analysis to compute a 'sentiment' score. It then plots these sentiment scores as rolling averages across a 7-day and 30-day period. Finally, it categorieses the sentiment scores into positive (sentiment < 0), neutral (sentiment = 0), and negative (sentiment < 0) classes to see how the two dictionary approaches compare. 


Optional parameters:
    -d  --data_path     <str>  Path to the csv file with headlines/tweets/textlines 
    -o  --output_path   <str>  Path to where the plots and files should be stored 

Usage: 

$ python3 src/sentiment_analysis.py

""" 

"""
=======================
Import the Dependencies
=======================

"""
#Operating system functionality 
import os
import argparse 

#Tools for working with data & visualisation 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

#SpacyTextBlob sentiment analysis 
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

#Add spacytextblob to the nlp pipeline
#We're using the English small library
nlp = spacy.load("en_core_web_sm")  
spacy_text_blob = SpacyTextBlob() 
nlp.add_pipe(spacy_text_blob) 

#VADER dependencies 
import nltk
nltk.download('vader_lexicon') 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# create a VADER sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

"""
==================
Argparse Arguments
==================

"""
# Initialize ArgumentParser class
ap = argparse.ArgumentParser()
    
# Argument 1: Path to the image directory
ap.add_argument("-d", "--data_path",
                type = str,
                required = False,
                help = "Path to the csv file with headlines/tweets/textlines",
                default = "data/subset_data.csv")

# Argument 2: Path to the output directory
ap.add_argument("-o", "--output_path",
                type = str,
                required = False,
                help = "Path to where the plots and files should be stored",
                default = "Output/")
    
# Parse arguments
args = vars(ap.parse_args()) 

"""
----------------
Define functions
----------------
"""
    
"""
SpacyTextBlob plots 
"""
# Smoothed sentiment plot for SpacyTextBlob 
#Build a function which creates a seaborn rolling average map 
def smoothed_sentiment_textblob(text_date_window, textblob_sentiment, output_path):
    
    #If the input data is weekly
    if text_date_window == "1-week":
        #Set the following seaborn asthetics 
        sns.set_style('darkgrid')
        plt.figure(figsize=(10,6), tight_layout=True)
        ax = sns.lineplot(data=textblob_sentiment,
                          x='publish_date', y='weekly',
                          label = "7 Day Average",
                          palette='pastel', 
                          linewidth=2.5) 
        ax.set(xlabel="Publish_date", ylabel = "Sentiment Score")
        ax.set(title = "SpacyTextBlob sentiment with a 7 day rolling average")
        ax.grid(color='grey', linestyle='-', linewidth=0.5)
        #save the plot 
        ax.figure.savefig(os.path.join(output_path, "TextBlob_weekly_sentiment.png"))
    
    else:
        #If the input data is monthly,
        #Set the following seaborn asthetics 
        sns.set_style('darkgrid')
        plt.figure(figsize=(10,6), tight_layout=True)
        ax = sns.lineplot(data=textblob_sentiment,                      
                          x='publish_date', y='monthly',
                          label = "30 Day Average",
                          palette='pastel', 
                          linewidth=2.5) 
        ax.set(xlabel="Publish_date", ylabel = "Sentiment Score")
        ax.set(title = "SpacyTextBlob sentiment with a 30 day rolling average")
        ax.grid(color='grey', linestyle='-', linewidth=0.5)
        #save the plot 
        ax.figure.savefig(os.path.join(output_path, "TextBlob_monthly_sentiment.png"))
            
"""
VADER plots 
"""
        
# Smoothed sentiment plot for VADER 
# Build a function which creates a seaborn rolling average map 
def smoothed_sentiment_vader(text_date_window, vader_sentiment, output_path):
    if text_date_window == "1-week":
        sns.set_style('darkgrid')
        plt.figure(figsize=(10,6), tight_layout=True)
        ax = sns.lineplot(data=vader_sentiment,
                          
                          x='publish_date', y='weekly',
                          label = "7 Day Average",
                          palette='pastel', 
                          linewidth=2.5) 
        ax.set(xlabel="Publish_date", ylabel = "Sentiment Score")
        ax.set(title = "VADER sentiment with a 7 day rolling average")
        ax.grid(color='grey', linestyle='-', linewidth=0.5)
        #save the plot 
        ax.figure.savefig(os.path.join(output_path, "VADER_weekly_sentiment.png"))
    
    else:
        sns.set_style('darkgrid')
        plt.figure(figsize=(10,6), tight_layout=True)
        ax = sns.lineplot(data=vader_sentiment,                        
                          x='publish_date', y='monthly',
                          label = "30 Day Average",
                          palette='pastel', 
                          linewidth=2.5) 
        ax.set(xlabel="Publish_date", ylabel = "Sentiment Score")
        ax.set(title = "VADER sentiment with a 30 day rolling average")
        ax.grid(color='grey', linestyle='-', linewidth=0.5)
        #save the plot 
        ax.figure.savefig(os.path.join(output_path, "VADER_monthly_sentiment.png"))
        
"""
=============
Main Function
=============

"""
def main():
    
    print("\nHello, let's conduct some sentiment analysis") 
    print("\nI'll just get us all set up") 
    
    """
    ------------------------------------------
    Create variables with the input parameters
    ------------------------------------------
    """
    data_path = args["data_path"]
    output_path = args["output_path"]
    
    #New users may want to create a new output directory, which we'll create here using the name defined in the terminal 
    #The code reads, "if an output path doesn't exist, please create one using os.mkdir()"
    if not os.path.exists(output_path):   
        os.mkdir(output_path) 
    
    #create 2 sets of the data, 1 for TextBlob and one for vader 
    textblob_data = pd.read_csv(data_path)
    vader_data = pd.read_csv(data_path)

    """
    ------------------------------------------
    SpaCy TextBlob Sentiment Analysis Pipeline 
    ------------------------------------------
    """
    
    #Let the user know the data will take a while to load 
    print("\nSetup complete - we'll start with some SpacyTextBlob Analysis") 
    
    """
    Get the sentiment scores
    """
    
    print("\nI'm about to start getting the sentiment score - this might take a few moments") 
    #Create an empty list to store sentiment scores 
    sentiment_tracker = []

    # For document headline in datafile
    # Group the headlines together in batches of 5000 (for efficiency) 
    for doc in nlp.pipe(textblob_data["headline_text"], batch_size = 5000):
        # calculate the sentiment of the headline
        sentiment = doc._.sentiment.polarity
        # append this to sentiment_tracker list
        sentiment_tracker.append(sentiment)
    
    print("\nSentiment's calculated, now we'll create the rolling average plots") 
    
    # append the sentiment_tracker list to the dataframe and save as output csv file
    textblob_data.insert(len(textblob_data.columns), "sentiment", sentiment_tracker)
    output_csv_path = os.path.join(output_path, "TextBlob_sentiment_tracker.csv")
    textblob_data.to_csv(output_csv_path, index = False)
        
    
    """
    Calculate the rolling averages 
    """
    
    # Create a sentiment dataframe with date as the index and sentiment scores to calculate means based on date
    textblob_sentiment = pd.DataFrame(
        {"sentiment": sentiment_tracker}, # create a column to hold sentiment scores 
        index = pd.to_datetime(textblob_data["publish_date"], format='%Y%m%d', errors='ignore'))
    
    #Create a column which holds the 7-day and 30-day rolling averages 
    textblob_sentiment['weekly'] = textblob_sentiment.sentiment.rolling(7).mean()
    textblob_sentiment['monthly'] = textblob_sentiment.sentiment.rolling(30).mean()
    
    
    """
    Plot the rolling averages 
    """
     
    smoothed_sentiment_textblob("1-week", textblob_sentiment, output_path)   #Weekly 
    
    print("\nI've got the weekly average plot, now I'll collect the monthly") 
    smoothed_sentiment_textblob("1-month", textblob_sentiment, output_path)  #Monthly 
    
    """
    Sentiment type comparison
    """
    # Create a new column which categorises sentiment into 'positive', 'neutral', and 'negative' for comparison
    textblob_sentiment['Sentiment_Type']=''
    textblob_sentiment.loc[textblob_sentiment.sentiment>0,'Sentiment_Type']='positive'
    textblob_sentiment.loc[textblob_sentiment.sentiment==0,'Sentiment_Type']='neutral'
    textblob_sentiment.loc[textblob_sentiment.sentiment<0,'Sentiment_Type']='negative'
    
    #Create and save a simple barchart 
    textblob_barplot = textblob_sentiment.Sentiment_Type.value_counts().plot(kind='bar',title="TextBlob Sentiment Analysis")
    textblob_barplot.figure.savefig(os.path.join(output_path, "TextBlob_barplot.png"))
   
    
    print("\n\nTextBlob analysis complete! Now we'll move on to VADER...") 
    
    
    """
    ---------------------------------
    VADER Sentiment Analysis Pipeline 
    ---------------------------------
    """
    
    print("\nI'm just about to get the VADER sentiment scores") 
        
    """
    Get the sentiment scores
    """
    
    # Create a new column in our vader_data with the sentiment scores from vader 
    vader_data['sentiment'] = vader_data['headline_text'].apply(lambda headline_text: sid.polarity_scores(headline_text))
    
    #Create a columm with just the compound value of the sentiment scores & create categories
    vader_data['compound'] = vader_data['sentiment'].apply(lambda score_dict: score_dict['compound'])
    vader_data['sentiment_type']=''
    vader_data.loc[vader_data.compound>0,'sentiment_type']='positive'
    vader_data.loc[vader_data.compound==0,'sentiment_type']='neutral'
    vader_data.loc[vader_data.compound<0,'sentiment_type']='negative'
    
        
    """
    Calculate the rolling averages 
    """
    
    print("\nVADER sentiment's calculated. Now we'll plot your rolling averages.") 
    
    #create a list from the compound column 
    vader_timeseries = vader_data["compound"].tolist()
    
    # Then create a dataframe with this list and the publish_date data converted into date form 
    vader_sentiment = pd.DataFrame(
    {"sentiment": vader_timeseries}, # create a column to hold sentiment scores 
    index = pd.to_datetime(vader_data["publish_date"], format='%Y%m%d', errors='ignore'))
    
    #Create a column which calculates the 7-day and 30-day rolling averages 
    vader_sentiment['weekly'] = vader_sentiment.sentiment.rolling(7).mean()
    vader_sentiment['monthly'] = vader_sentiment.sentiment.rolling(30).mean()
    
    """
    Plot the rolling averages 
    """
     
    smoothed_sentiment_vader("1-week", vader_sentiment, output_path)   #Weekly 
    
    print("\nI've got the weekly average plot, now I'll collect the monthly") 
    
    smoothed_sentiment_vader("1-month", vader_sentiment, output_path)  #Monthly 
    
    
    """
    Sentiment type comparison
    """
    
    #We already have the sentiment column categorised so we can jump straight into plotting it 
    vader_barplot = vader_data.sentiment_type.value_counts().plot(kind='bar',title="VADER Sentiment Analysis")
    vader_barplot.figure.savefig(os.path.join(output_path, "VADER_barplot.png"))
    
    
    print(f"\nThat's you all complete! You can find the plots in your {output_path} folder.\n\n") 
    
    
if __name__=="__main__":
    #execute main function
    main()    

    
