#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

text = []
rating = []

text_file = open("./text/bad.txt", "r")
unit = text_file.readlines()
text = text + unit
rating = rating + ['bad'] * len(unit)

text_file = open("./text/good.txt", "r")
unit = text_file.readlines()
text = text + unit
rating = rating + ['good'] * len(unit)

text_file = open("./text/medium.txt", "r")
unit = text_file.readlines()
text = text + unit
rating = rating + ['medium'] * len(unit)

print(len(text))
print(len(rating))

outcome = pd.DataFrame(list(zip(text, rating)), columns=['text', 'rating'])
outcome.to_csv('./text/text.csv', index=False)
