#python3 get_tweet_dump.py  ## takes more than >8 hrs to run
spark-submit trend_preprocessing.py bitcoin_data.csv preprocess/twitter_trend
spark-submit preprocess_tweets.py Bitcoin_tweets.csv daily_tweets_v2.csv preprocess/twitter_sentiment
spark-submit twitter_model.py preprocess prediction
cp prediction prediction_intermediate
#spark-submit prediction_analysis.py prediction prediction_final

