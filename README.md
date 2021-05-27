[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![describtor - e.g. python version](https://img.shields.io/badge/Python%20Version->=3.6-blue)](www.desired_reference.com) ![](https://img.shields.io/badge/Software%20Mac->=10.14-pink)

# Sentiment Analysis 

## Exploring Australian headlines with SpacyTextBlob and VADER


<div align="center"><img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/emotions%20(1).png"/></div>




Sentiment is defined in the Oxford Dictionary as ‘a view or opinion which is held or expressed’ (Oxford Dictionary, 2020). In the world of computational linguistics, sentiment stays true to this definition. Here, it is used to determine the general views, opinions, and emotions held within a text by analysing how positive or negative the language held within is. It is a popular and useful tool for getting an idea of how opinions change over time or for comparing the general mood between texts. For this reason, it is common to see sentiment analysis used within the world of cultural data science for exploring questions such as “How do opinions on the corona vaccine differ between countries?” or “Are news articles today more ‘doom and gloom’ than they were 20 years ago?”. 

In this first assignment, we look at two of the most popular dictionary based approaches to sentiment analysis, SpacyTextBlob and VADER. We'll run the same data through each library and extract the sentiment scores to compare how each will perform. 


## Table of Contents 

- [Assignment Description](#Assignment)
- [Scripts and Data](#Scripts)
- [Methods](#Methods)
- [Operating the Scripts](#Operating)
- [Discussion of results](#Discussion)

## Assignment Description

In this assignment we were using sentiment analysis to monitor headlines from Australian news channel ABC. The data file used contains around 1.2 million headlines which were collected over an 18-year timespan, from February 2003 until the end of December 2020.  


___The assignment asked us to complete the following steps:___ 

i)	  Calculate the sentiment score for every headline in the data using a dictionary-based approach such as TextBlob

ii)	  Create and save a plot of sentiment over time with a 1-week rolling average 

iii)  Create and save a plot of sentiment over time with a 1-month rolling average 

iv)	  Ensure that there are clear values on the x and y axis, a title, and a legend. 

v)	  Write a short summary describing what the 2 plots show 

___Personal focus of the assignment:___

I took this assignment as an opportunity to firstly compare different dictionary-based sentiment analysis approaches. I did this by adding VADER sentiment analysis into the assignment and comparing the two approaches using bar plots of the number of sentiment categories detected by the two approaches from the same dataset. I then wanted to focus on the visualisation of the plots and so I experimented with many of the functionalities in matplotlib and seaborn to create more visually pleasing line plots. Finally, I developed the Notebook assignment into a python script which could be run from the terminal. 


## Scripts and Data 

***Data***

The data used throughout the assignmend has been taken from Kaggle's million headlines dataset. If the user wishes to use the full dataset, they will need to download and upload the file to their working directory, due to the large file size. This can be downloaded from [here](https://www.kaggle.com/therohk/million-headlines![image](https://user-images.githubusercontent.com/52678852/119842145-b7d79500-bf06-11eb-8926-e425374bcad0.png)). Alternatively, a subsection of the data can be found in the data folder, which is also possible to run on the scripts. 

***Scripts***

There is one script to be run for this assignment, which can be found under the name of 'sentiment_analysis.py' in the src folder. The notebook has also been included for a more visual exploration through the various stages of the script. 


## Methods 

The assignment was completed by conducting the TextBlob analysis, then the VADER analysis, and then comparing the outputs of each approach. Before diving into the steps taken, let's take a moment to look at what these two dictionary approaches are: 

 <img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/ASPACY.png" alt="alt text" width="100" height="40">


**SpaCyTextBlob** (TextBlob) is built on top of the NLTK package and returns two properties for a given sentence, paragraph, or document. These are: 

1. **Polarity**: A measure of how negative or positive the text is, running along a scale from -1 to +1. 0 is considered neurtral while anything above is considered positive in sentiment while anything below is considered negative.

2. **Subjectivity**: A float number between 0 and 1 which indicates how subjective or objective a sentence is. Sentences with a high subjectivity score refer to a personal opinion, emotion, or judgement while sentences with low subjectivity are objective or refer to factual information.

TextBlob uses a method of ‘averaging’ whereby it ignores the syntax and structure of the words and simply computes the average value for the entire sentence. 

### VADER 🤓 

VADER is one of the better known approaches to sentiment analysis and particularly good at handling social media data. This is because it not only tells you the overall sentiment of the sentence, but also _how_ positive/negative the sentence is in general. This helps to avoid confusion in sentences such as "oh you're awfully sweet", which is clearly meant in an endearing way but could end with a low sentiment score if taken at face value. To get around this problem, VADER assesses 4 sentiment metrics: 

1. Positivity ((proportion of the sentence which is positive) 
2. Neutrality (proportion of the sentence which is neutral)
3. Negativity (proportion of the sentence which is negative)
4. Compound (the sum of all the lexicon ratings which have been standardised to fall between -1 to + 1)

The lexicon of words used by VADER is somewhat more limited than other dictionary approaches, meaning not all words are assigned a sentiment score. The total values are therefore based on the words which are included only, which can be hard to decipher in large bodies of text. 

### Steps taken for the analysis

Creating the line plots: 
1.	Create a list of sentiment scores by looping over the datafile, extracting the polarity score of each headline, and appending this into the list. 
2.	The sentiment scores were added as a new column to the datafile and saved. 
3.	A sentiment data frame was then created with date as the index and sentiment as a column to calculate the means based on the date.
4.	A ‘weekly’ column was added to calculate the 7-day rolling average for that date 
5.	A ‘monthly’ column was added to calculate the 30-day rolling average for that date 
6.	A plot function was then build to take time window, data source, and output path to create a seaborn line plot which is saved into the output folder. 

***Creating comparison plots:*** 

To compare the performance of each analysis method, the ‘sentiment’ column of each was categorised into 3 sentiment categories, positive (for scores above 0), neutral (for scores of 0) and negative (or scores below 0). This categorisation could generalise across the two dictionary approaches because both TextBlob’s polarity sentiment and VADER’s ‘compound’ sentiment are on the same scale from -1 to 1. Boxplots were used to visualise how close the sentiment distribution would be across the three categories for each method. A similar distribution would indicate the two methods performed similarly with the same data while a considerable difference would indicate why it is important to consider which sentiment analysis is being used for the task at hand. 


## Operating the Scripts 

***1. Clone the repository***

The easiest way to access the files is to clone the repository from the command line using the following steps 

```bash
#clone repository
git clone https://github.com/Orlz/Sentiment_Analysis.git

```

***2. Download the data from Kaggle*** (_Optional_) 

The data can be downloaded from Kaggle [here](https://www.kaggle.com/therohk/million-headlines![image](https://user-images.githubusercontent.com/52678852/119842145-b7d79500-bf06-11eb-8926-e425374bcad0.png))

***3. Create the virtual environment***

You'll need to create a virtual environment which will allow you to run the script using all the relevant dependencies. This will require the requirements.txt file attached to this repository. 


To create the virtual environment you'll need to open your terminal and type the following code: 

```bash
bash create_virtual_environment.sh
```
And then activate the environment by typing:

```bash
$ source language_analytics01/bin/activate
```

***4. Run the Script***

There is just one script to run for this assignment (sentiment_analysis.py) with 2 optional parameters: 

```
Letter call  | Parameter      | Required? | Input Type     | Description                                         
-----------  | -------------  |--------   | -------------  |                                                     
`-d`         | `--data_path`  | No        | String         | Path to where the data is stored                     
`-o`         | `--output_path'| No        | String         | Path to where the output should be stored    
```

The script can be run from the commandline with the following code: 

```bash
$ python3 src/sentiment_analysis.py
```       

## Discussion of Results 

**Left:  TextBlob,     Right: VADER**

 <img src="https://github.com/Orlz/Sentiment_Analysis/blob/main/Output/TextBlob_monthly_sentiment.png" alt="alt text" width="450" height="250">  <img src="https://github.com/Orlz/Sentiment_Analysis/blob/main/Output/VADAR_monthly_sentiment.png" alt="alt text" width="450" height="250"> 

Comparing the two sentiment dictionaries certainly opens up for some interesting results. SpaCy TextBlob has a generally stable sentiment which hovers around 0.2 throughout the 18-year time span, while VADER paints a more depressing picture with sentiments hovering around -0.05. VADER also displays more deviation in its wave pattern. In both plots, the sentiment follows a typical wavelike form, following each other in trend by remaining rather flatline for the first 10 years with a gradual increase between 2012 to 2015 and falling between the years of 2018 to 2020. VADER seems to detect the increase beforehand, with sentiment begining to rise much earlier, around 2008, and falling more significantly. Surprisingly, sentiment in both approaches seems to be on the increase towards the end of 2020 - could it be that this when news of the vaccine startied hitting the headlines? In terms of comparison, it is comforting to see similar patterns reflected in both methods, albeit with different baseline sentiments. This brings us to the next point.... how different are those baseline scores? 

 <img src="https://github.com/Orlz/Sentiment_Analysis/blob/main/Output/TextBlob_barplot.png" alt="alt text" width="400" height="300">  <img src="https://github.com/Orlz/Sentiment_Analysis/blob/main/Output/VADER_barplot.png" alt="alt text" width="400" height="300"> 


We see that SpaCy TextBlob has a much more neutral baseline than VADER. This would suggest that for a headline to be considered ‘overly positive’ or ‘overly negative’ would have to be a lot more polarised for SpaCy TextBlob to detect it than VADER. On the other hand, we can see why VADER’s sentiment hovered below 0, with such a high proportion of negative headlines. It seems to pull apart the headlines and place them more at the extremes. Simple mechanics could explain these differences, such as VADER being especially focused on social media which may have a generally higher sentiment score than headlines, or that the dictionary values for common words may greatly differ between the two approaches. The important thing is that each approach was able to capture the same trends and displayed similar patterns across the 18 years of Australian news headlines. We can conclude that it is worth considering the baseline value for each dictionary or running two dictionary approaches simultaneously when considering sentiment analysis to ensure the method used best reflects the data at hand. 


___Teaching credit___ Many thanks to Ross Deans Kristiansen-McLachlen for providing an interesting and supportive venture into the world of Language Analytics! 

<div>Icons made by <a href="https://www.freepik.com" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
