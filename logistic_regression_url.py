import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import tldextract

digits = ['0', '1', '2', '3', '4', '5', '6' '7', '8', '9']
vowels = ['a', 'e', 'i', 'o', 'u']

def digit_count(url):
    count = 0
    for c in url:
        if c in digits:
            count += 1
    return count

def vowel_count(url):
    count = 0
    for c in url:
        if c in vowels:
            count += 1
    return count

def url_length(url):
    return len(url)

def dot_count(url):
    count = 0
    for c in url:
        if c == '.':
            count += 1
    return count

def ext_pos_tld():
    f_pos = open('pos.txt', 'r')
    pos_urls = [url.strip() for url in f_pos.readlines()]

    m = {}
    for url in pos_urls:
        tld = tldextract.extract(url)
        suffix = tld.suffix
        if not suffix in m:
            m[suffix] = 1
        else:
            continue
    pos_tlds = [k for k, v in m.items()]
    return pos_tlds

def ext_dga_tld():
    f_dga = open('dga.txt', 'r')
    dga_urls = [url.strip() for url in f_dga.readlines()]

    m = {}
    for url in dga_urls:
        tld = tldextract.extract(url)
        suffix = tld.suffix
        if not suffix in m:
            m[suffix] = 1
        else:
            continue
    dga_tlds = [k for k, v in m.items()]
    return dga_tlds

def get_pos_tld(url):
    tld = tldextract.extract(url)
    if tld.suffix in pos_tlds:
        return 1
    else: 
        return 0

def get_dga_tld(url):
    tld = tldextract.extract(url)
    if tld.suffix in dga_tlds:
        return 0
    else:
        return 1


def feature_engineering(url, pos_tlds, dga_tlds):
    l = url_length(url)
    pd = dot_count(url)
    pos_test = get_pos_tld(url)
    dga_test = get_dga_tld(url)
   

    x = [l, vowel_count(url)/(l*1.0), digit_count(url)/(l*1.0), pd, pos_test, dga_test]
    return x



f_right = open('pos.txt', 'r')

right_urls = [url.strip() for url in f_right.readlines()]

right_urls = right_urls[: 25000]

f_dga = open('dga.txt', 'r')

dga_urls = [url.strip() for url in f_dga.readlines()]

dga_urls = dga_urls[: 25000]

if __name__ == '__main__':

    pos_tlds = ext_pos_tld()
    dga_tlds = ext_dga_tld()

    X = []
    y = []
    x_test = []
    y_test = []
    
    for i in range(0, 20000):
        X.append(feature_engineering(right_urls[i], pos_tlds, dga_tlds))
        y.append(1)

    for i in range(0, 20000):
        X.append(feature_engineering(dga_urls[i], pos_tlds, dga_tlds))
        y.append(0)

    for i in range(20001, 20101):
        x_test.append(feature_engineering(right_urls[i], pos_tlds, dga_tlds))
        y_test.append(1)

    for i in range(20001, 20101):
        x_test.append(feature_engineering(dga_urls[i], pos_tlds, dga_tlds))
        y_test.append(0)

    X = np.array(X)
    y = np.array(y)
    x_test = np.array(x_test)

    clf = LogisticRegression()
    
    clf.fit(X, y)
    
    y_pred = clf.predict(x_test)

    count = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            count += 1

    print('precision %f ' % (count*1.0 / len(y_test)))
    print(clf)
