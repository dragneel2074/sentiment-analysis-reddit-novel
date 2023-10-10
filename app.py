import os
import praw
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import spacy
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud

REDDIT_CLIENT_ID = "lI0C_W9_eESoiS2mtUMNDg"
REDDIT_CLIENT_SECRET = "IK1Vn7s0EZGiNt6vMZ54sfT6pYvbHA"
REDDIT_USERNAME="Tiger_in_the_Snow"

reddit = praw.Reddit(
   client_id = REDDIT_CLIENT_ID,
   client_secret = REDDIT_CLIENT_SECRET,
   user_agent = f"script:sentiment-analysis:v0.0.1 (by {REDDIT_USERNAME})"
)
sub = reddit.subreddit('noveltranslations')
query = "The Legendary Mechanic"
results = sub.search(query, limit=10)

stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

total_positive = 0
total_negative = 0
total_comments = 0
comments_for_cloud = []

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]    
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [stemmer.stem(i) for i in text]
    return ' '.join(text)

def tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

# Load pre-trained Roberta tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('aychang/roberta-base-imdb') 
model = RobertaForSequenceClassification.from_pretrained('aychang/roberta-base-imdb',num_labels=2)

# Add classification head 
model.classifier = torch.nn.Linear(768, 2) 

results_list = list(results)

if not results_list:
     print("No results found for the provided query.")
else:
    for submission in results_list:
        print(submission.title)

        submission.comments.replace_more(limit=None)
        all_comments = submission.comments.list()

        for comment in all_comments:
            comment_body = comment.body
            #print(comment_body)

            text = transform_text(comment_body)
            comments_for_cloud.append(text)
            if text:  # Ensuring the transformed text is not empty
                tokens = tokenize(text)
                
                # Tokenize comments
                tokenized_input = tokenizer(tokens, return_tensors='pt', truncation=True, padding=True)

                # Get sentiment predictions 
                outputs = model(**tokenized_input)
            
                probabilities = torch.softmax(outputs.logits, dim=-1)  # Convert logits to probabilities
                mean_probabilities = probabilities.mean(dim=1)  # Calculate the mean probability over all tokens in a sequence

                # Calculate the percentage for both classes
                positive_pct = mean_probabilities[0][1].item() * 100
                negative_pct = mean_probabilities[0][0].item() * 100

                print(f"Positive: {positive_pct}% , Negative: {negative_pct}%")

                total_positive += positive_pct
                total_negative += negative_pct
                total_comments += 1

# Calculate the average percentages
if total_comments > 0:
    avg_positive = total_positive / total_comments
    avg_negative = total_negative / total_comments

    print(f"\nAverage sentiment - Positive: {avg_positive}% , Negative: {avg_negative}%")
else:
    print("No Comments Found for Sentiment Analysis")

if total_comments > 0:

    all_comments_string = ' '.join(comments_for_cloud)

    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords.words('english'), 
                    min_font_size = 10, max_words=30).generate(all_comments_string)

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0)
    plt.title("Word Cloud of Comments")
    plt.show()

    # Plot sentiment bar graph
    labels = ['Positive Sentiment', 'Negative Sentiment']
    values = [avg_positive, avg_negative]

    plt.bar(labels, values, color=['green', 'red'])
    plt.xlabel('Sentiment')
    plt.ylabel('Average Percentage')
    plt.title('Average Sentiment Analysis of Comments')
    plt.show()
