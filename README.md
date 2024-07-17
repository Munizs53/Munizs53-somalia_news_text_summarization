# somalia_news_text_summarization
his project focuses on developing an automatic news text summarization system for the Somali language using natural language processing (NLP) techniques.his project focuses on developing an automatic news text summarization system for the Somali language using natural language processing (NLP) techniques. The system preprocesses Somali news articles, applies the TextRank algorithm with cosine similarity on TF-IDF vectors for summarization, and generates concise summaries. The goal is to facilitate access and comprehension of news content, save time and effort, and support researchers, journalists, policy framers, and general users. 
 Data Description  
For the experimentation process on this project, there is a dataset of 100 articles 
borrowed from different Somali news sources. The articles represent what has been used 
as the basis for the development and testing of the automated text summarization 
system. The selection targeted fair representation of topics and styles, hence able to 
afford a firm base for establishing viability of the implemented summarization 
techniques.  Some of the applied pre-processing steps, like tokenization, stop-word 
elimination, and stemming, prepare the text for the application of the TextRank 
algorithm. This would then create summaries that are both dense and with pertinent 
information.   
Corpus Preparation  
This dataset was downloaded from Caasimada Online, one of the key online news 
websites based in Somalia. For this reason, the dataset had to be versatile, containing 
many topics and writing styles, so 100 articles were chosen. These were to constitute an 
all-encompassing foundation to develop an accordingly tested summarization system. 
Obtaining these articles began with the collection of articles in their native digital 
format. After this, the text was pre-processed to make it analytically eligible.  
It tokenized using breaking up texts into singular words or phrases, removed stop words 
that are very common and thus insignificant, and provided stemming to bring the words 
down to their simplest forms. These preprocessing steps were all aimed at making the 
text optimum for feeding into the TextRank algorithm since it needs clear and concise 
input when summaries with a really high accuracy level are to be done.  
Therefore, processing the corpus through these steps had readied the dataset for 
summarization to be able to work on it and come up with definitive informative 
summaries of Somali articles.   
 Somali Stop Words Corpus   
stop words are those classes of words that are very common and have very little effect 
on the meaning within the text. Therefore, in text analysis, for efficiency's sake, such 
stop words get removed at this pre-processing level. A corpus supply with 500 stop 
words for Somali was put together for use in this current project.  
It was identified and collated those stop words to keep away some of these unnecessary 
or non-essential words within the text, hence allowing summarization to focus on big 
and meaningful content. To achieve a dataset that was more fine-tuned and target 
oriented for subsequent processes, such as tokenization and stemming, the stop words 
were already removed beforehand. The table below, 3.1, illustrates some of the 500 
Somali stop words used in this project.  

Implementation  
The system was developed after the installation of NLTK with the help of Python 3.7.4. 
The set-up of Python 3.7.4 along with NLTK was installed comfortably. Some libraries 
were included in the program, such as BeautifulSoup, nltk, and re. The rest two are used 
to include cosine_similarity and TfidfVectorizer from sklearn.metrics.pairwise and 
sklearn.feature_extraction.text, respectively. These two tools were used while 
developing the system and running the experiments also. This section is broken down 
into marked-off sections explaining how the methodology will get the following piece 
of work done.  
Phase I: Preprocessing  
Preprocessing techniques in the text summarization process essentially have a major 
role in preparing the data for further processing. The various techniques that are 
basically introduced at this stage include tokenization, removal of stop words, 
stemming, and handling the length of sentences.  
Sentence Segmentation and Tokenization  
Sentence segmentation is a process for breaking down text documents into individual 
sentences and identifying their boundaries. After identification of the sentence-ending 
punctuations the period (.), question mark (?), exclamation mark (!) the system will 
partition the text documents with respect to the sentences.  
Tokenization of a document means breaking the input document into words. This step 
is attained through spaces, commas (), and special symbols between words. To be sure, 
in this step, the frequency of every word is calculated and saved for further use. One 
problem that arises with sentence segmentation is settling on when a period truly is to 
be considered as a full stop or to be part of an abbreviation. These periods in "e.g." or 
"Dr." may trigger a segmentation error. For this, the researcher mitigated through the 
use of a list of abbreviations previously compiled by other researchers.  
 
stop word removal refers to the removal of common words that do not add important 
meaning to the text. Using a list of 500 Somali stop-words, he did this so as to be left 
with meaningful words remaining for analysis.  
Those might include stop words like articles, conjunctions, prepositions, pronouns, or 
some other type of word that tends to show up in most sentences.  
In the case of Somali, stop words such as "ku" meaning to, "iyo" meaning and, "ka" 
meaning from are removed because their removal would not have any effect on the 
meaning of the document even if removed. As it works, this finds a match within the 
list of stop words in the database for the input document. The list of stop words used in 
this work.   
 Phase II: TextRank Algorithm Implementation  
Another popular technique used in NLP in text summarization is the TextRank 
algorithm, applying aspects of the PageRank algorithm for web search. The TextRank 
algorithm has subsequently been applied to score the sentences in a ranked order of 
importance following Phase I. Implementation details for the application of the 
TextRank algorithm are given in the following subsections.  
Sentence Graph Construction  
In the TextRank algorithm, a graph is created in which nodes denote the sentences of a 
document. The edges are drawn between nodes, based on the similarity of the sentences. 
Basically, once two sentences are compared, the similarity is measured by the cosine 
similarity between their word vectors.  
Run TextRank Algorithm  
First, a graph of sentences needs to be generated. Then, implement the TextRank 
algorithm for updating an importance score of every sentence. Then, iteratively update 
all scores for each node by scores of all other nodes - its neighbors in this graph.  
 Phase III: Summary Generation  
The last step will be to summarize by selecting the best-ranked sentences picked out 
through the TextRank algorithm. These sentences go to merge for clear and concise 
summarization, hence giving refined input of the original Somali news articles to users.  
 Training the model  
We begin our Somali news text summarization model training by the dataset 
preparation.  
Our dataset consists of 100 articles crawled from Caasimada Online, which is basically 
one of the biggest news websites in the Somali language. Considering them as raw data, 
this present corpus encompasses a wide range of topics and differs in length. By way of 
preparation, cleaning of the text precedes all other. Specifically, we remove undesired 
characters, punctuation marks, and numbers to create by bandwidth for standardization 
of text across all articles. This initial cleaning step helps our model to make sure about 
not wasting time on any useless content while it learns on it. Once that the data is 
cleaned, we proceed with its preprocessing.  
This stage involves several essential tasks to enhance the quality of our text data.  
We use NLTK for word tokenization and sentence tokenization. The text is thereby split 
into words and sentences correspondingly. In the last stage, we remove stop words, 
generally common words not adding value to the meaning of a text. In Somali, this 
would filter out some 500 preset stop words. After stopping these words, lemmatization 
is used to reduce words in basic forms so that same vocability is used across an article. 
This preprocessing pipeline readies our data for the next step a crucial phase of model 
28   
training. Now that we have preprocessed data, let us implement TextRank and produce 
a summary for all articles. TextRank is a graph-based algorithm; it assigns an 
importance score to individual sentences based on their relations with other sentences 
in an article.  
It uses the cosine similarity on TF-IDF vectors to construct a similarity matrix that 
quantifies the similarity between pairs of sentences.  
It, however, further ranks these sentences iteratively with respect to these scores in order 
to get important sentences to be part of the summary. In this way, we are able to generate 
concise summaries that ensure essential information is retained from every article by 
only selecting sentences with very high scores. This TextRank application therefore 
makes our model capable of summarizing Somali news articles effectively. Somalian 
News Articles Summaries Extraction is an approach to extractive summarization in two 
major steps: Original and generation. 
3 Evaluation and Discussion  
Performance in this "NLP for Somalian News Text Summarization" project was 
assessed based on preprocessing, the construction of the TextRank similarity matrix, 
and the execution of the TextRank algorithm. ROUGE scores were obtained as a 
performance metric, quantifying the overlap of n-grams and the longest common 
subsequence between the system-generated summaries and the reference summaries. 
Additionally, human evaluators rated the summaries' relevance and in formativeness 
using precision, recall, and F1-score. Human evaluation was considered essential for 
assessing relevance and coherence, as human judgment is the gold standard.   
For the human evaluation, we randomly sampled ten articles from our dataset. Three human 
evaluators identified the three most important sentences in each article. The model was then 
trained to generate summaries by selecting the top three sentences, and ROUGE scores were 
computed to compare the human-generated summaries with the model-generated summaries.
The scores suggest that the system does well with ROUGE-1, ROUGE-2 and ROUGE-L 
scores showing large n-gram overlap between the extracted summaries and human summaries. 
This indicates that the system is capable of capturing important information from the articles. 
There were no major issues with the relevance, coherence or conciseness for which human
evaluation provided positive scores meaning that they were informative as well as readable. 
However, some important details are occasionally missing (or there is lack of logical flow) in 
parts within such summaries.
Conclusions  
This project investigated a news text summarization system of the Somali news articles 
through the application of NLP techniques using Python. The proposed system had an 
objective to design a prototype of extractive summarization by applying a pre-processing 
function for constructing the TextRank similarity matrix and generating concise summaries 
through the subjects of TextRank algorithms.  
A preprocessing function was implemented to clean the text and prepare it with 500 Somali 
stop words to reduce noise. Afterwards, applying TextRank using cosine similarity on TF-IDF 
vectors builds a similarity matrix from which key sentences in the articles can be identified. 
Summaries generated were then evaluated using ROUGE scores and human evaluation 
metrics: relevance, coherence, and conciseness.  
