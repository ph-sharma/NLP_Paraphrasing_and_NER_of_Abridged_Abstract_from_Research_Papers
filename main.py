# Importing Libraries
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import PyPDF2


# Extracting Abstract from Paper
def extract_abstract(pdf_file):
    reader = PyPDF2.PdfFileReader(pdf_file)
    num_pages = reader.numPages
    for page_num in range(num_pages):
        page_text = reader.getPage(page_num).extractText()
        abstract_start = page_text.find("Abstract")
        abstract_end = page_text.find("Keywords")
        if abstract_start != -1 and abstract_end != -1:
            return page_text[abstract_start:abstract_end]
    return None


pdf_path = 'paper.pdf'
abstract = extract_abstract(pdf_path)
text = abstract

print("Extracted Text: \n", text)

# tokenizing the text and removing stop words from them
stopWords = set(stopwords.words("english"))
words1 = word_tokenize(text)
words = [word for word in words1 if not word in stopWords]

# creating frequency chart
freqTable = dict()
for word in words:
    word = word.lower()
    if word in stopWords:
        continue
    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1

# Sentence tokenization
sentences = sent_tokenize(text)
sentenceValue = dict()

for sentence in sentences:
    for word, freq in freqTable.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

sumValues = 0
for sentence in sentenceValue:
    sumValues += sentenceValue[sentence]

try:
    average = int(sumValues / len(sentenceValue))
except ZeroDivisionError:
    average = 0

summary = ''
for sentence in sentences:
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
        summary += " " + sentence
print("=======================================================================================")
print("\nSummary:\n-----------")
print(summary)


nlp = spacy.load("en_core_web_sm")
s1 = nlp(text)
s2 = nlp(summary)

# printing similarity between original text and summary
print("\nSimilarity: ", s1.similarity(s2))

doc = nlp(summary)
for ent in doc.ents:
    print(ent.text, " | ", ent.label_, " | ", spacy.explain(ent.label_), "\n")

# Getting Synonyms to replace with the phrases in the text
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))

    return list(synonyms)


# Paraphrasing Text based on synonyns on wornet from nltk corpus
def paraphrase_text(text1):
    tokens = word_tokenize(text1)
    paraphrasedText = []
    for token in tokens:
        synonyms = get_synonyms(token)
        if synonyms:
            paraphrasedText.append(synonyms[0])
        else:
            paraphrasedText.append(token)
    return ' '.join(paraphrasedText)


paraphrased_text = paraphrase_text(summary)
print("Original Text:\n", summary, "\n")
print("Paraphrased Text:\n", paraphrased_text)