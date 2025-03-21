#from eod import EodHistoricalData
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import twint
#import nest_asyncio
#nest_asyncio.apply()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
import random
import time
import sys

output = sys.argv[1]


## Code referred from internet.

def getTweets(search_term, until, since, limit=20):
    """
    Configures Twint and returns a dataframe of tweets for a specific day.
    """
    # Configuring Twint for search
    c = twint.Config()

    # The limit of tweets to retrieve
    c.Limit = limit

    # Search term
    c.Search = search_term

    # Removing retweets
    c.Filter_retweets = True
    
    # Popular tweets
    #c.Popular_tweets = True
    
    # Verified users only
    #c.Verified = True

    # Lowercasing tweets
    c.Lowercase = True

    # English only
    c.Lang = 'en'

    # Tweets until a specified date
    c.Until = until + " 00:00:00"
    
    # Tweets since a specified date
    c.Since = since + " 00:00:00"
    
    # Making the results pandas friendly
    c.Pandas = True
    
    # Stopping print in terminal
    c.Hide_output = True

    # Searching
    try:
        twint.run.Search(c)
    except twint.token.RefreshTokenException:
        time.sleep(60)
    #twint.run.Search(c)
    
    # Assigning the DF
    df = twint.storage.panda.Tweets_df
    
    # Returning an empty DF if no tweets were found
    if len(df)<=0:
        return pd.DataFrame()
    
    # Formatting the date
    df['date'] = df['date'].apply(lambda x: x.split(" ")[0])
    
    # Returning with english filter to account for an issue with the twint language filter
    return df[df['language']=='en']
  
 ## Running the twint query
def tweetByDay(start, end, df, search, limit=20):
    """
    Runs the twint query everyday between the given dates and returns
    the total dataframe. 
    
    Start is the first date in the past.
    
    End is the last date (usually would be current date)
    """
    # Finishing the recursive loop
    if start==end:
        # Removing any potential duplicates
        df = df.drop_duplicates(subset="id")
        print(len(df))
        return df    
    
    # Appending the new set of tweets for specified window of time
    tweet_df = getTweets(
        search, 
        until=(datetime.strptime(start, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d"), 
        since=start, 
        limit=limit
    )
    
    # Running the query a few more times in case twint missed some tweets
    run = 0 
    
    while len(tweet_df)==0 and run<=2:
        
        # Running query again
        tweet_df = getTweets(
            search, 
            until=(datetime.strptime(start, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d"), 
            since=start, 
            limit=limit
        )
        
        # Counting how many times it ran
        run += 1
    
    # Adding the new tweets
    df = df.append(tweet_df, ignore_index=True)
    
    # Updating the new start date
    new_start = (datetime.strptime(start, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
    # Printing scraping status
    print(f"\t{len(df)} Total Tweets collected as of {new_start}\t")
    
    # Running the function again
    return tweetByDay(
        start=new_start, 
        end=end, 
        df=df, 
        search=search
    )

# Getting tweets daily

search=['#bitcoin', 
    '#cryptocurrency', 
    '#crypto', 
    '#btc', 
    '#bitcoinmining',
    '#bitcoinnews', 
    '#bitcoins',
    '#bitcointrading',
    '#cryptotrading',
    '#binance', 
    '#cryptonews', 
    '#bitcoincash']


tweet_df=[]
for s in search:
    df = tweetByDay(
        start="2021-01-01", 
        end="2022-12-07", 
        df=pd.DataFrame(), 
        search=s, 
        limit=100
    )
    tweet_df.append(df)

# Saving file for later use
tweet_df=pd.concat(tweet_df).drop_duplicates(subset=['tweet'])
tweet_df=tweet_df[['date','tweet']].drop_duplicates()
tweet_df.columns=['date','text']
tweet_df.to_csv(output+"/tweets_daily_2.csv", index=False)
