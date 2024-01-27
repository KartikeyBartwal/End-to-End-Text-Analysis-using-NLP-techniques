import numpy as np
import pandas as pd
import re
import logging
import os
from colorlog import ColoredFormatter
import nltk
nltk.download('averaged_perceptron_tagger')  
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from nltk.corpus import cmudict
nltk.download('cmudict')
from math import ceil



class TextAnalysis:
    text = ""
    words = []
    PositiveWords = []
    StopWords = []
    
    def __init__(self , text , PositiveWords , NegativeWords , StopWords):
        self.text =  text
        self.words = re.findall(r'\w+\b' , text)
        self.PositiveWords = PositiveWords
        self.NegativeWords = NegativeWords
        self.StopWords = StopWords
    
    def Positive_Score(self):
        positive_words_count = 0
        stopwords_count = 0
        Total_Words = len(self.words)
        
        print(f"Total Words to Process: {Total_Words}")
        
        for word in self.words:
            # word should not be a stopword
            if word in self.StopWords:
                print("Stopword: ", word)
                stopwords_count += 1
                continue
            
            if word in self.PositiveWords:
                print("Positive Word: ", word)
                positive_words_count += 1
            else:
                print("Normal Word: ", word)
        
        print()
        print(f"Positive words found: {positive_words_count}")
        print(f"StopWords words found: {stopwords_count}")
        print(f"Total Words: {Total_Words}")
        
        
        positive_percentage = (positive_words_count / Total_Words) * 100
        print(f"Positive words percentage: {positive_percentage}%")
        
        positive_percentage = "{:.2f}".format(positive_percentage)
        return positive_percentage
        
    
    def Negative_Score(self):
        negative_words_count = 0
        stopwords_count = 0
        Total_Words = len(self.words)
        
        print(f"Total Words to Process: {Total_Words}")
        
        for word in self.words:
            # word should not be a stopword
            if word in self.StopWords:
                print("Stopword: ", word)
                stopwords_count += 1
                continue
            
            if word in self.NegativeWords:
                print("Negative Word: ", word)
                negative_words_count += 1
            else:
                print("Normal Word: ", word)
        
        print()
        print(f"Negative words found: {negative_words_count}")
        print(f"StopWords words found: {stopwords_count}")
        print(f"Total Words: {Total_Words}")
        
        
        negative_percentage = (negative_words_count / Total_Words) * 100
        print(f"Positive words percentage: {negative_percentage}%")
        
        negative_percentage = "{:.2f}".format(negative_percentage)
        return negative_percentage

    
    def Polarity_Score(self):
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(self.text)
    
    def Subjectivity_Score(self):
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(self.text)
        subjectivity_score = scores['compound']
        return subjectivity_score
    
    def AvgSentence_Length(self):
        curr_length = 0
        arr = []

        for char in text:
            if(char == '.'):
                arr.append(curr_length)
                curr_length = 0
            else:
                curr_length += 1
        
        arr.append(curr_length)
        
        sum = 0
        for length in arr:
            print(length , end = " ")
            sum += length
        print()
        
        return sum / len(arr)
    
    def Percentage_Complex_Words(self):
        complex_word_count = self.Complex_Word_Count(False)
        Total_Words = len(self.words)
        
        return ((complex_word_count) / Total_Words) * 100
    
    def Fog_Index(self):
        word_count = len(self.words)
        def num_sentences(text):
            full_stop_count = 0
            for character in text:
                if(character == '.'):
                    full_stop_count += 1
            
            return full_stop_count + 1
        sentence_count = num_sentences(self.text)
        complex_words_count = self.Complex_Word_Count(False)
        
        
        print("Formula: ")
        print('''
            FogIndex= 0.4 * ((word_count / sentence_count) + 100 * (complex_words_count / word_count))
        ''')
        
        return 0.4 * ((word_count / sentence_count) + 100 * (complex_words_count / word_count))
            
    
    def Avg_Words_Per_Sentence(self):

        words_in_sentence = []
        curr_word_count = 0
        curr_word = ""

        for char in text:
            # '.' => move to the next word
            if(char == '.'):
                curr_word_count += 1
                words_in_sentence.append(curr_word_count)
                curr_word_count = 0
                curr_word = ""
            # current word ends
            elif(char == ' '):
                curr_word_count += 1
                curr_word = ""
        
            # ad the character to the current word
            else:
                curr_word += char
        
        words_in_sentence.append(curr_word_count)
        
        sum = 0
        for word_count in words_in_sentence:
            print(word_count , end = " ")
            sum += word_count
        
        print()
        
        return sum / len(words_in_sentence)
    
    def Complex_Word_Count(self , logs = True):
        def is_complex_word(word):
            # Get part-of-speech tags for the word
            tagged = nltk.pos_tag([word])  
            pos = tagged[0][1]
            # Check if the word has certain parts of speech that might indicate complexity      
            return pos.startswith('JJ') or pos.startswith('NN') or pos.startswith('VB')  # Checking for adjectives, nouns, or verbs

        complex_word_count = 0
        for word in self.words:
            if(is_complex_word(word)):
                complex_word_count += 1
                
                if(logs):
                    print("Complex Word:" , word)
        
        print("Total Words:" , len(self.words))
        print("Complex Words: " , complex_word_count)
        
        return complex_word_count

    
    def Word_Count(self):
        return len(self.words)
    
    def Syllable_Per_Word(self):
        d = cmudict.dict()
        def count_syllables(word):
            if word.lower() in d:
                return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
            else:
                # If word is not found in the dictionary, provide an alternate solution
                # This might not be completely accurate for all words
            
                return max([len([y for y in x if y[-1].isdigit()]) for x in d if x.lower().startswith(word.lower())], default=1)
        
        sum = 0
        count = 0
        for word in self.words:
            syllable_count = count_syllables(word)
            sum += syllable_count
            
            print(word ,":" , syllable_count)
            count += 1
        
        return ceil(sum / count)
    
    def Personal_Pronouns(self):
        personal_pronouns = [
    'I', 'me', 'my', 'mine', 'myself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves']
        personal_pronoun_count = 0
        for word in self.words:
            if(word in personal_pronouns):
                print("Personal pronoun:" , word)
                personal_pronoun_count += 1
        print()
        print("Total Words:" , len(self.words))
        print("Number of Personal Pronouns:" , personal_pronoun_count)
        
        return personal_pronoun_count
        
    
    def Avg_Word_Length(self):
        sum_of_characters = 0
        for word in self.words:
            print(word , len(word))
            sum_of_characters += len(word)
            
        print("Sum of Characters: " , sum_of_characters)
        print("Number of Words: " , len(self.words) )
        return sum_of_characters//len(self.words)