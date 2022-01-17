'''
PROGRAMMING ASSIGNMENT 2: Word Sense Disambiguation (WSD) by Team 4: AIT-580 under Dr. Liao

Introduction :
Authors : Muhammad Hassan
Date : 06/24/2021
WSD is a technique in NLP and ontology. This refers to identification of in which sense a word is
used in a sentence, when the word has several meanings.
This python file, decision-list.py, act as a decision list classifier to perform WSD by identifying features from
training data. Further log-likelihood probabilities of both senses is used to calculate the likelihood of the learned
decision rules.

Features Used: words, word indices upto 8 position on either directions and created a decision list by learning train data

Sample my-decision-list:

['-1_word_telephone', 6.149747119504682, 'phone']
['-1_word_access', 4.954196310386876, 'phone']
['-1_word_car', -4.247927513443585, 'product']
['-1_word_end', 4.08746284125034, 'phone']
['1_word_dead', 3.700439718141092, 'phone']
['-1_word_computer', -3.700439718141092, 'product']
['-1_word_came', 3.700439718141092, 'phone']
['-1_word_ps2', -3.700439718141092, 'product']
['-7_word_telephone', 3.700439718141092, 'phone']
['-1_word_gab', 3.4594316186372978, 'phone']

Instructions to run decision-list.py :

1) If nltk stop words library is not installed, download nltk stop words using nltk.download('stopwords')
2) Run the decision-list.py program in the command prompt as follows:
$ python decision-list.py line-train.xml line-test.xml my-decision-list.txt > my-line-answers.txt
3) A text file with all decisions learned by the program will be stored in my-decision-list.txt and output text file
with list of answer instances and sense id can be found my-line-answers.txt.
Also, any output file name can be specified since STDOUT function is used.

Sample Output :

my-line-answers.txt

<answer instance="line-n.w8_059:8174:" senseid="phone"/>
<answer instance="line-n.w7_098:12684:" senseid="phone"/>
<answer instance="line-n.w8_106:13309:" senseid="phone"/>
<answer instance="line-n.w9_40:10187:" senseid="phone"/>
<answer instance="line-n.w9_16:217:" senseid="product"/>
<answer instance="line-n.w8_119:16927:" senseid="product"/>
<answer instance="line-n.w8_008:13756:" senseid="product"/>
<answer instance="line-n.w8_041:15186:" senseid="phone"/>
<answer instance="line-n.art7} aphb 05601797:" senseid="phone"/>
<answer instance="line-n.w8_119:2964:" senseid="product"/>


Algorithm :

Step1: Scrap the training and test datasets from the files
Step2: Preprocess the data (remove stopwords and punctuation)
Step3: Use CFD (Conditional Frequency Distribution) for calculating the frequencies of the learned rules from the training dataset
Step4: Use CPD(Conditional Probability Distribution) for calculating the probabilities of the frequencies
Step5: Use logarithm of ratio of probabilities of both senses to calculate the likelihood each decision rule
Step6: Determine the majority sense from training data
Step7: Perform predicitons on the test dataset based on the learned decision rules
Step8: Post the answers from the prediction to STDOUT
Step9: Write the decision rules to a file

Sample Output from scorer(with word indices 8 on either side of the word. This gives optimal output) :
Baseline accuracy is 57.14285714285714%
Accuracy after adding learned features is 85.71428571428571%
Confusion matrix is
Predicted  phone  product
Actual
phone         60        6
product       12       48

'''

import nltk, string, re, math, sys
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import ELEProbDist
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# command line arguments for the file sources of training data, testing data, decision list
training_data = sys.argv[1]
testing_data = sys.argv[2]
my_decision_list = sys.argv[3]

# Word under analysis (Ambiguous word)
root_word = "line"

# initialize an empty list, to be used as the decision list.
decision_list = []

def process_text(input_text):
    """
    Function to perform pre-processing and return the processed text

    :param input_text: input text to be preprocessed
    :return: the preprocessed text

    """
    input_text = input_text.lower()

    # removing the standard stop word from the text
    stop_words = stopwords.words("english")
    stop_words.extend(string.punctuation)

    # treating "lines" and "line" as a single entity
    input_text = input_text.replace("lines", "line")
    corpus = [re.sub(r'[\.\,\?\!\'\"\-\_/]', '', w) for w in input_text.split(" ")]
    corpus = [w for w in corpus if w not in stop_words and w != '']
    return corpus


def get_n_word(n, context):
    """
    Function to get a word at a certain index from the root_word

    :param n: relative index
    :return: the word at relative index n from the root_word. Empty string if there is no such word present

    """
    root_index = context.index(root_word)
    n_word_index = root_index + n
    if len(context) > n_word_index and n_word_index >= 0:
        return context[n_word_index]
    else:
        return ""


# Function to add a new "condition" learned from the training data to the decision list
def add_word_cond(cfd, data, n):

    """
    Function to add a new "condition" learned from the training data to the decision list
    :param cfd: the conditional frequency ditribution
    :param data: the training data
    :param n: relative index n from the root_word
    :return: the calculated conditional frequency distribution

    """

    for element in data:
        sense, context = element['sense'], element['text']
        n_word = get_n_word(n, context)
        if n_word != '':
            condition = str(n) + "_word_" + re.sub(r'\_', '', n_word)
            cfd[condition][sense] += 1
    return cfd



def calculate_log_likelihood(cpd, rule):

    """
    Function to return the log of ration of sense probabilities

    :param cpd:
    :param rule:
    :return: the word at relative index n from the root_word. Empty string if there is no such word present

    """
    prob = cpd[rule].prob("phone")
    prob_star = cpd[rule].prob("product")
    div = prob / prob_star
    if div == 0:
        return 0 # consider log-likelihood as zero since log(0) is undefined
    else:
        return math.log(div, 2)



def check_rule(context, rule):
    """
    Function to check whether the rule is satisfied in a given context

    :param context: context under consideration
    :param rule: rule under consideration
    :return: the boolean response of whether the rule is satisfied in a given context.

    """
    rule_scope, rule_type, rule_feature = rule.split("_")
    rule_scope = int(rule_scope)

    return get_n_word(rule_scope, context) == rule_feature


def predict(context, majority_label):
    """
    Function to predict and return the predicted sense on test data.

    :param context: context under consideration
    :param majority_label:
    :return: the predecited sense on test data.

    """
    for rule in decision_list:
        if check_rule(context, rule[0]):
            if rule[1] > 0:
                return ("phone", context, rule[0])
            elif rule[1] < 0:
                return ("product", context, rule[0])
    return (majority_label, context, "default")


def scrap_from_train_file(file):
    """
    Function to extract text from training data through XML parsing.

    :param file: input training data file
    :return : the scraped training data as an array

    """
    with open(training_data, 'r') as data:
        soup = BeautifulSoup(data, 'html.parser')
        train_data = []
        for instance in soup.find_all('instance'):
            sntnc = dict()
            sntnc['id'] = instance['id']
            sntnc['sense'] = instance.answer['senseid']
            text = ""
            for s in instance.find_all('s'):
                text = text + " " + s.get_text()
            sntnc['text'] = process_text(text)
            train_data.append(sntnc)
        return train_data

def scrap_from_test_file(file):
    """
    Function to extract text from tes data through XML parsing.

    :param file: input test data file
    :return : the scraped testing data

    """
    with open(testing_data, 'r') as data:
        test_soup = BeautifulSoup(data, 'html.parser')

        test_data = []
        for instance in test_soup('instance'):
            sntnc = dict()
            sntnc['id'] = instance['id']
            text = ""
            for s in instance.find_all('s'):
                text = text + " " + s.get_text()
            sntnc['text'] = process_text(text)
            test_data.append(sntnc)
        return test_data

#Use BeautifulSoup to scrap from training file.
train_data = scrap_from_train_file(training_data)

# Use conditional frequency distribution to add learned rules to the decision list.
# Here word indices upto 8 position on either directions are used to form the decision list.
cfd = ConditionalFreqDist()
cfd = add_word_cond(cfd, train_data, 1)
cfd = add_word_cond(cfd, train_data, -1)
cfd = add_word_cond(cfd, train_data, 2)
cfd = add_word_cond(cfd, train_data, -2)
cfd = add_word_cond(cfd, train_data, 3)
cfd = add_word_cond(cfd, train_data, -3)
cfd = add_word_cond(cfd, train_data, 4)
cfd = add_word_cond(cfd, train_data, -4)
cfd = add_word_cond(cfd, train_data, 5)
cfd = add_word_cond(cfd, train_data, -5)
cfd = add_word_cond(cfd, train_data, 6)
cfd = add_word_cond(cfd, train_data, -6)
cfd = add_word_cond(cfd, train_data, 7)
cfd = add_word_cond(cfd, train_data, -7)
cfd = add_word_cond(cfd, train_data, 8)
cfd = add_word_cond(cfd, train_data, -8)

# Initialize a Condition probability distribution to calculate the probabilities of the frequencies recorded above
cpd = ConditionalProbDist(cfd, ELEProbDist, 2)

# store the learned rules into the decision list
for rule in cpd.conditions():
    likelihood = calculate_log_likelihood(cpd, rule)
    decision_list.append([rule, likelihood, "phone" if likelihood > 0 else "product"])
    #sort the decision list according to probabilities to decrease the execution time
    decision_list.sort(key=lambda rule: math.fabs(rule[1]), reverse=True)

# extracting the test data through XML parsing
test_data  = scrap_from_test_file(testing_data)

# Calculating the frequencies of each senses
senseA, senseB = 0.0, 0.0
for element in train_data:
    if element['sense'] == "phone":
        senseA += 1.0
    elif element['sense'] == 'product':
        senseB += 1.0
    else:
        print("warning no match")

# Calculating the majority sense
majority_sense = "phone" if senseA > senseB else "product"

# Performing the predictions
predictions = []
for element in test_data:
    pred, _, r = predict(element['text'], majority_sense)
    id1 = element['id']
    predictions.append(f'<answer instance="{id1}" senseid="{pred}"/>')
    print(f'<answer instance="{id1}" senseid="{pred}"/>')

# Storing the decision list into a file
with open(my_decision_list, 'w') as output:
    for listitem in decision_list:
        output.write('%s\n' % listitem)
