# NLP-projrcts
This code is a Python script that performs sentiment analysis on tweets obtained from a JSON file. It uses several libraries including Pandas, NLTK, TextBlob, and NumPy.

The script performs the following steps:

Loads a JSON file containing tweets data.
Converts the JSON data into a Pandas DataFrame.
Preprocesses the tweets to remove stop words, lemmatize words, and remove punctuation.
Identifies the most frequently occurring words in the preprocessed text using Pandas Series and DataFrame objects and NumPy.
Selects the most frequent words occurring at least 100 times and uses them to further preprocess the tweets.
Extracts bigrams from the final preprocessed text using NLTK.
Performs sentiment analysis on the bigrams using TextBlob.
Categorizes the sentiment as positive, negative, or neutral and saves the results to a CSV file.
The script contains comments throughout to explain each step in detail. Additionally, it saves two intermediate CSV files during the process: one with the preprocessed data before bigram extraction and sentiment analysis, and one with the results of the second objective.

The code assumes that the input file is called "tweets.json" and that it is located in the same directory as the script. It saves the resulting CSV files in the same directory as the script.

To use the script, the user must have Python 3 installed as well as the required libraries. The NLTK library also requires the user to have downloaded the stopwords corpus and the WordNetLemmatizer.

Overall, this script provides an example of how to perform sentiment analysis on text data and extract meaningful insights from it.



