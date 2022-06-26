#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------->  准备工作 ---------------------------------------------

import time
import random
import requests
import jieba
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from imageio import imread
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
# -----------------------------------------------------------------------------


# -------------------->  爬取设置 ---------------------------------------------
def getHtml(url, headers):

    try:
        r = requests.get(url, timeout=30, headers=headers)
        r.raise_for_status()
        return r.text
    except:
        return ''

# 获取评论
def getComment(html):
    soup = BeautifulSoup(html, 'html.parser')
    comments_list = []  # 评论列表
    comment_nodes = soup.select('.comment > p')
    for node in comment_nodes:
        comments_list.append(node.get_text().strip().replace("\n", "") + u'\n')
    return comments_list

# 浏览器信息 - 依据特定电脑信息（https://blog.csdn.net/ysblogs/article/details/88530124）
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
           'Cookie': 'll="108288"; bid=Qm5EziAHieA; __utma=30149280.2109946700.1655390050.1655390050.1655390050.1; __utmc=30149280; __utmz=30149280.1655390050.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmt=1; dbcl2="195546593:MS//bJ9SeK4"; ck=8G20; ap_v=0,6.0; push_noty_num=0; push_doumail_num=0; __utmv=30149280.19554; __utmb=30149280.8.9.1655390350573'}

depth = 30
# -----------------------------------------------------------------------------


# -------------------->  爬取 ---------------------------------------------
text = []
rating = []
for index in [2295163]:
    for rate in ['h', 'm', 'l']:
        print(index)
        print(rate)
        pre_url = "https://book.douban.com/subject/" + str(index) + "/comments/?percent_type=" + rate + "&"
        for i in range(1, depth):
            print('开始爬取第{}页评论...'.format(i))
            url = pre_url + 'start=' + str(20 * (i-1)) + '&limit=20&status=P&sort=new_score'
            html = getHtml(url, headers)
            content = getComment(html)
            if content:
                text.append(content)
                rating.append(rate)
            # 设置随机休眠防止IP被封
            time.sleep(1 + float(random.randint(1, 20)) / 20)
    print('成功完成爬取任务')
print(len(text))
print(len(rating))
outcome = pd.DataFrame(list(zip(text, rating)), columns=['text, rating'])
outcome.to_csv('.text/text.csv')
print('-------------------------finish-----------------------------')
# -----------------------------------------------------------------------------
