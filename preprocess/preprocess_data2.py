import numpy as np
from collections import Counter


def cross(a, b):
    set_a = set(a)
    set_b = set(b)
    return list((set_a.union(set_b)) ^ (set_a ^ set_b))

def main():
    spam = Counter()
    ham = Counter()
    with open("../data/SMSSpamCollection", encoding='utf-8') as f:
        line = f.readline()
        while line:
            words = line.replace("\t", " ").replace("\n", " ").split(" ")
            line = f.readline()
            if words[0] == "spam":
                spam.update(words[1:])
                # for word in words[1:]:
                #     spam.update(word)
            else:
                ham.update(words[1:])
                # for word in words[1:]:
                #     ham.update(word)
    print(spam.most_common(10))
    print(ham.most_common(10))

if __name__ == "__main__":
    main()
