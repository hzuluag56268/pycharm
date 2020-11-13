'''# Remove whitespaces and print the result
movie_no_space = movie_lower.strip("$")         .lstrip  .rstrip

# Split the string into substrings and print the result
movie_split = movie_no_space.split()


# Split the string using commas and print results
movie_no_comma = movie_tag.split(",")

# Join back together and print results
movie_join = ' '.join(movie_no_comma)

# Split string at line boundaries
file_split = file.splitlines()

Find if a pattern occurs between the characters 1 and 4 (inclusive) of
string using
string.find(pattern, 1, 5)
.
If not found, .find() will return -1.
z
Replace old for new in string using string.replace(old, new).


plan = {    "field": courses[0], "tool": courses[1]}

# Complete the placeholders accessing elements of field and tool keys
my_message = "If you are interested in {dat[field]}," \
             " you can take the course related to {dat[tool]}"

# Use dictionary to replace placeholders
print(my_message.format(dat=plan))

print(f" {field1!r} in the {fact1}st century")
!r=printable notation, :e=exponetial, :.2f


from string import Template
# Create a template
wikipedia = Template("$tool is a $description")

# Substitute variables in template
print(wikipedia.substitute(tool=tool1, description=description1))








import re

# Find all matches of regex
print(re.findall(r"@robot\d\W", 'The string where to look for the regex'))


regex = r"@robot\d\W"

# Find all matches of regex
print(re.findall(regex, sentiment_analysis))

 Replace the regex_sentence with a space
sentiment_sub = re.sub(regex_sentence, " ", sentiment_analysis)


Matchanycharacter(exceptnewline):  .

re.findall(r"^the\s\d+s", my_string)  ^ #Only checks at the beginning of the string
Only checks at the End of the string: $

OR operator re.findall(r"Elephant|elephant", my_string
OR operator re.findall(r"[a-zA-Z]+\d", my_string)

Set of characters: [ ]^ transforms the expression to negative
my_links = "Bad website: www.99.com. Favorite site: www.hola.com"
re.findall(r"www[^0-9]+com", my_links


 we used the .match() method.
The reason is that we want to match the pattern from the beginning of the string.

we used the .search() method. It checks if the regex is Anywhere in the string given,
Not only at the beginning like .match()

re.sub(r"<.+?>", "", string)
? Makes the expression None greedy

r"(hate|dislike|disapprove).+?(?:movie|concert)\s(.+?)\."
( | | ) Used to group and capture using or operand  ,(?:) used to group but not capture

text = "Python 3.0 was released on 12-03-2008."
information = re.search('(\d{1,2})-(\d{2})-(\d{4})', text)
information.group(3)

Give a name to groups using r"(?P<name>regex)
text = "Austin, 78701"
cities = re.search(r"(?P<city>[A-Za-z]+).*?(?P<zipcode>\d{5})", text)
cities.group("city")  #Only possible with .search or .match
'Austin

Using numbered capturing groups to reference back
sentence = "I wish you a happy happy birthday!"
re.findall(r"(\w+)\s\1", sentence)
['happy']
#The number one is referencing to the first group,
# its copying the first group once more

sentence = "Your new code number is 23434. Please, enter 23434 to open the door."
re.findall(r"(?P<code>\d{5}).*?(?P=code)", sentence)
['23434']
#Again back referencing But with the name of the group

re.sub(r"(?P<word>\w+)\s(?P=word)", r"\g<word>", sentence)
#For  substituting uses g instead of P

Positive lookahead (?=) makes sure that first part of the expression
is followed by the lookahead expression.
    Positive lookbehind (?<=) returns all matches
that are preceded by the specified pattern.
    re.findall(r"\w+(?=\spython)",


negative lookahead (?!) and negative lookbehind (?<!)


''' # Brief summary of regular expressions


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from collections import Counter

# Set pandas columns to display at max width
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None)
# Set seaborn aesthetic features to pretty up our plots
sns.set()


# Read the csv and assign to the DataFrame 'wine_df'
wine_df = pd.read_csv('https://github.com/datacamp/working-with-text-data-in-python-live-training/blob/master/data/wine_reviews.csv?raw=true')

# View the first five rows
print(wine_df.head())
print(wine_df.info())


wine_df['designation'].fillna(wine_df['winery'] + ' - unknown', inplace=True)
print(wine_df['designation'].sample(5))
print('')

wine_df['location'] = wine_df['region']+ ' - ' +wine_df['country'].str[:3].str.upper()


wine_df['variety'] = wine_df['variety'].str.strip()\
                                .str.lower().str.replace('\s{2,}', ' ', regex=True)

'''
text = ' '.join(wine_df['description']).lower()




stopwords1 = ['a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'all',
 'also',
 'am',
 'an',
 'and',
 'any',
 'are',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 "can't",
 'cannot',
 'com',
 'could',
 "couldn't",
 'did',
 "didn't",
 'do',
 'does',
 "doesn't",
 'doing',
 "don't",
 'down',
 'drink',
 'during',
 'each',
 'else',
 'ever',
 'few',
 'flavor',
 'flavors',
 'for',
 'from',
 'further',
 'get',
 'had',
 "hadn't",
 'has',
 "hasn't",
 'have',
 "haven't",
 'having',
 'he',
 "he'd",
 "he'll",
 "he's",
 'her',
 'here',
 "here's",
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 "how's",
 'however',
 'http',
 'i',
 "i'd",
 "i'll",
 "i'm",
 "i've",
 'if',
 'in',
 'into',
 'is',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'k',
 "let's",
 'like',
 'me',
 'more',
 'most',
 "mustn't",
 'my',
 'myself',
 'no',
 'nor',
 'not',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'otherwise',
 'ought',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 'palate',
 'r',
 'same',
 'shall',
 "shan't",
 'she',
 "she'd",
 "she'll",
 "she's",
 'should',
 "shouldn't",
 'since',
 'so',
 'some',
 'such',
 'than',
 'that',
 "that's",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 "there's",
 'these',
 'they',
 "they'd",
 "they'll",
 "they're",
 "they've",
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 'very',
 'was',
 "wasn't",
 'we',
 "we'd",
 "we'll",
 "we're",
 "we've",
 'were',
 "weren't",
 'what',
 "what's",
 'when',
 "when's",
 'where',
 "where's",
 'which',
 'while',
 'who',
 "who's",
 'whom',
 'why',
 "why's",
 'wine',
 'with',
 "won't",
 'would',
 "wouldn't",
 'www',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves']
stopwords1 = stopwords1 + ["flavor", "flavors", "wine", "drink", "palate"]

tokens = [word for word in word_tokenize(text) if word.isalpha()]
final_words = [word for word in tokens if word not in stopwords1 ]


word_count = Counter(final_words)
for word in word_count.most_common(10):
    print(word[0] + ': ' + str(word[1]) + ' mentions')

#ree = re.findall('tan\w{2,4}', ' '.join(word_count.keys()))

print(word_count['tannin'])
print(word_count['nose'])

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

lemmatized_words = [stemmer.stem(wor) for wor in final_words ]
lemmatized_count = Counter(lemmatized_words)
for word in lemmatized_count.most_common(10):
    print(word[0] + ': ' + str(word[1]) + ' mentions lemma')
print('')
stemmed_words = [stemmer.stem(word) for word in final_words ]
stemmed_count = Counter(stemmed_words)
for word in stemmed_count.most_common(10):
    print(word[0] + ': ' + str(word[1]) + ' mentions stemmed')
print('')

lemmatized_count_df = pd.DataFrame(lemmatized_count.most_common(10),
                                   index= None,
                                   columns=['word', 'freq'] )
print(lemmatized_count_df)

colors = [*reversed(['#ffffd9','#ffffd9','#edf8b1','#c7e9b4','#7fcdbb'
                     ,'#41b6c4','#1d91c0','#225ea8','#253494','#081d58'])]
''' #part1
'''
plt.bar('word','freq', data=lemmatized_count_df, color=colors)
plt.xticks(rotation=45)
plt.show()
'''  #plt bar
'''
sns.set_palette(sns.color_palette('cividis', 10))
sns.barplot('freq', 'word', data=lemmatized_count_df,ci=None)
plt.show()

''' # sns bar


oak_filter = wine_df['description'].str.contains('oak', case=False)

wine_df_oak = wine_df.loc[oak_filter]
pd.set_option('display.max_colwidth', None)
oak_pattern = r"(\s|^)[Oo]ak(iness|y|s)?[/.,\s]"
oak_filter_re_df = wine_df[wine_df['description'].str.contains(oak_pattern)]
#print(oak_filter_re_df.head())
grouped = oak_filter_re_df.groupby('variety').count()['country'].sort_index(ascending=False)
no_grouped = wine_df.groupby('variety').count()['country'].sort_index(ascending=False)
ratio_series = (grouped * 100 / no_grouped).sort_values(ascending=False)

sns.barplot( no_grouped, no_grouped.index,ci=None, color='lightsalmon')
sns.barplot( grouped, grouped.index,ci=None, color='firebrick')
plt.show()
sns.barplot( ratio_series, ratio_series.index,ci=None, color='firebrick')
plt.show()