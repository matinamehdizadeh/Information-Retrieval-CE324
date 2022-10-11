from selenium import webdriver
import time
import json
import numpy as np
import pandas as pd
from tkinter import *
from PIL import ImageTk, Image
import random

e1 = 0
e2 = 0


def crawl(path):   
    start = open(path, "r")
    links = []
    for line in start:
        stripped_line = line.strip()
        line_list = stripped_line.split()
        links.append(line_list[0])
    start.close()
    
    driver = webdriver.Chrome()
    check = []
    counter = 0
    idnum = 0
    allD = []
    titles = []
    driver.get('https://academic.microsoft.com/paper/2950893734')
    time.sleep(2)
    
    while(True):
        #print(counter)
        doc = {}
        docId = links[idnum].split("/")[4]
        if docId not in check:
            check.append(docId)
            counter += 1
    
        else:
            idnum += 1
            continue
        driver.get(links[idnum])
        idnum += 1
        time.sleep(2)
        try:
            doc["id"] = docId 
            
            title = driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/h1').text
            if(title in titles):
                counter -= 1
                continue
            doc["title"] = title
            titles.append(title)
            
            abstract = driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/p').text
            doc["abstract"] = abstract
            
            date = driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/a/span[1]').text
            doc["date"] = date 
        
            authors = []
            author = driver.find_element_by_xpath('//div[@class="authors"]').text.split(",")
            for i in range(len(author)):
                r = author[i].split()
                s = ''
                for i in range(len(r)):
                    if(r[i].isnumeric() == False):
                        s += r[i] + ' '
                if(s[-1] == ' '):
                    s = s[:len(s)-1]
                authors.append(s)
            doc["authors"] = authors
            
            related_topics = driver.find_element_by_xpath('//div[@class="tag-cloud"]').text.split("\n")
            if(len(related_topics) > 3):
                related_topics = related_topics[0:len(related_topics)-1]
            doc["related_topics"] = related_topics
            
            citation_count = driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/div[1]/ma-statistics-item[2]/div[2]/div[2]/div[1]').text
            doc["citation_count"] = citation_count 
            
            reference_count = driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/div[1]/ma-statistics-item[1]/div[2]/div[2]/div[1]').text
            doc["reference_count"] = reference_count 
        except:
            counter -= 1
            continue
        references = []
        for i in range(11):
            try:
                reference = driver.find_element_by_xpath(f'//*[@id="mainArea"]/router-view/router-view/ma-edp-serp/div/div[2]/div/compose/div/div[2]/ma-card[{i+1}]/div/compose/div/div[1]/a[1]')
                url = reference.get_attribute("href")
                links.append(url)
                ids = url.split("/")[4]
                references.append(ids)
            except:
                break
        doc["references"] = references 
        allD.append(doc)
        if counter == 2000:
            break
    driver.close()
    with open('CrawledPapers.json', mode='w', encoding='utf-8') as f:
        json.dump(allD, f, indent = 2)


def pageRank(path, alpha):
    file = open(path, "r") 
    doc = json.load(file) 
    p = np.zeros((2000, 2000))
    nref = np.full((1, 2000), 1/2000)
    ids = []
    for i in range(2000):
        ids.append(doc[i]['id'])
    
    for i in range(2000):
        ref = doc[i]['references']
        validRef = []
        for j in range(len(ref)):
            if ref[j] in ids:
                validRef.append(ids.index(ref[j]))
        if(len(validRef) == 0):
            p[i] = nref
        else:
            for j in range(len(validRef)):
                p[i][validRef[j]] = 1/len(validRef)
    
    for i in range(2000):
        for j in range(2000):
            p[i][j] = (1-alpha) * p[i][j] + alpha/2000
    
    for i in range(100):
        nref = np.matmul(nref, p)
    rank = {}    
    
    for i in range(2000):
        rank[ids[i]] = nref[0][i]
    with open('CrawledPapers2.json', mode='w', encoding='utf-8') as f:
        json.dump(rank, f, indent=2)
 
        
def HITS(path, n):
    file = open(path, "r") 
    doc = json.load(file) 
    
    authors = []
    ids = []
    for i in range(2000):
        a = doc[i]['authors']
        ids.append(doc[i]['id'])
        for j in range(len(a)):
            if(a[j] not in authors):
                authors.append(a[j])
        
    A = np.zeros((len(authors), len(authors)))

    for i in range(2000):
        author = doc[i]['authors']
        references = doc[i]['references']
        m = []
        for ref in references:
            if(ref in ids):
                a = doc[ids.index(ref)]['authors']
                for j in range(len(a)):
                    m.append(authors.index(a[j]))
        for a in author:
            for index in m:
                A[authors.index(a)][index] = 1
    Aa = np.matmul(np.transpose(A), A)   
    a = np.full((1, len(authors)), 1)[0]
    for i in range(5):
        a = np.matmul(Aa, a)
        a = a/a.sum()
    ind = np.argpartition(a, -n)[-n:]
    ind = ind[np.argsort(a[ind])]
    topA = {}
    for i in range(len(ind)):
        s = str(authors[ind[i]]) + " has authority " +  str(a[ind[i]])
        r = str(len(ind) - i)
        topA[r] = s
    with open('CrawledPapers3.json', mode='w', encoding='utf-8') as f:
        json.dump(topA, f, indent=2)
                  


def ContentBased(vec, path):
    
    df = pd.read_csv("data.csv")
    df = df.replace(np.nan, 0)
    col = df.columns.tolist()
    docV = np.zeros((2000, len(col)))
    file = open(path, "r") 
    doc = json.load(file) 
    for i in range(2000):
        related = doc[i]['related_topics']
        for r in related:
            for j in range(len(col)):
                if col[j] in r:
                    docV[i][j] = 1
    vec = df[vec:vec+1].values[0]
    cos_simularity = []
    for i in range(2000):
        dist = 1 - np.dot(vec, docV[i]) / np.sqrt(np.dot(docV[i], docV[i]) * np.dot(vec, vec))
        if np.isnan(dist):
            cos_simularity.append(2)
        else:
            cos_simularity.append(dist)
    cos_simularity = np.array(cos_simularity)
    ind = np.argpartition(cos_simularity, 10)[:10]
    ind = ind[np.argsort(cos_simularity[ind])]
    topA = {}
    for i in range(len(ind)):
        s = "page with id " + str(doc[ind[i]]['id']) + " is relevent with cos similarity " +  str(cos_simularity[ind][i])
        r = str(i+1)
        topA[r] = s
    with open('CrawledPapers4.json', mode='w', encoding='utf-8') as f:
        json.dump(topA, f, indent=2)


def CollaborativeFiltering(x, n, path):
    df = pd.read_csv("data.csv")
    df = df.replace(np.nan, 0)
    col = df.columns.tolist()
    docV = np.zeros((2000, len(col)))
    file = open(path, "r") 
    doc = json.load(file) 
    for i in range(2000):
        related = doc[i]['related_topics']
        for r in related:
            for j in range(len(col)):
                if col[j] in r:
                    docV[i][j] = 1
    vec = df[x:x+1].values[0]
    cos_simularity = []
    for i in range(len(df)):
        vecS = df[i:i+1].values[0]
        dist = 1 - np.dot(vec, vecS) / np.sqrt(np.dot(vecS, vecS) * np.dot(vec, vec))
        cos_simularity.append(dist)
    cos_simularity = np.array(cos_simularity)
    ind = np.argpartition(cos_simularity, (n+1))[:n+1]
    ind = ind[np.argsort(cos_simularity[ind])]
    newV = []
    for i in range(n+1):
        newV.append(df[ind[i]:ind[i]+1].values[0])
    newV = np.array(newV)
    vec = []
    for i in range(newV.shape[1]):
        vec.append(np.mean(newV[:, i]))
    cos_simularity = []
    for i in range(2000):
        dist = 1 - np.dot(vec, docV[i]) / np.sqrt(np.dot(docV[i], docV[i]) * np.dot(vec, vec))
        if np.isnan(dist):
            cos_simularity.append(2)
        else:
            cos_simularity.append(dist)
    cos_simularity = np.array(cos_simularity)
    ind = np.argpartition(cos_simularity, 10)[:10]
    ind = ind[np.argsort(cos_simularity[ind])]
    topA = {}
    for i in range(len(ind)):
        s = "page with id " + str(doc[ind[i]]['id']) + " is relevent with cos similarity " +  str(cos_simularity[ind][i])
        r = str(i+1)
        topA[r] = s
    with open('CrawledPapers5.json', mode='w', encoding='utf-8') as f:
        json.dump(topA, f, indent=2)
     
        
def CompleteM(st, etha):
    st = 100
    etha = 0.001
    df = pd.read_csv("data.csv")
    df = df.replace(np.nan, 0)
    test = np.zeros((len(df), 34))
    train = np.zeros((len(df), 34))
    
    for i in range(len(df)):
        vec = df[i:i+1].values[0]
        for j in range(len(vec)):
            if vec[j] != 0:
                rand = random.uniform(0, 1)
                if rand <= 0.8:
                    train[i][j] = vec[j]
                else:
                    test[i][j] = vec[j]
    
    u, s, v = np.linalg.svd(train, full_matrices=False)
    k = 2
    p = []
    q = []
    for i in range(len(u)):
        p.append(u[i][:2].tolist())
    for i in range(k):
        q.append(v[:,i].tolist())
    p = np.array(p)
    q = np.array(q)
    for i in range(st):
        for j in range(len(p)):
            dif = sum(train[j, :][train[j, :] != 0] - np.dot(p[j, :], q)[train[j, :] != 0])
            for kN in range(k):
                p[j, kN] = p[j, kN] - etha * 2 * (p[j, kN] * dif)
        for j in range(len(q)):
            dif = sum(train[:, j][train[:, j] != 0] - np.dot(p,q[:, j])[train[:, j] != 0])
            for kN in range(k):
                q[kN, j] = q[kN, j] - etha * 2 * (q[kN, j] * dif)
    
    newT = np.matmul(p, q)[0]
    er = np.power((test - newT)[test != 0] , 2).sum()
    outP = {}
    outP["error"] = "the error is: " + str(er)
    with open('CrawledPapers6.json', mode='w', encoding='utf-8') as f:
            json.dump(outP, f, indent=2)     
        
     
        
     
        
     
        
     
        
def inable():
    can.delete('all')
    can.create_image(0, 0, image = bg, anchor = "nw")
    but1_w = can.create_window(220, 70, anchor = "nw", window = but1)
    but2_w = can.create_window(220, 110, anchor = "nw", window = but2)
    but3_w = can.create_window(220, 150, anchor = "nw", window = but3)
    but4_w = can.create_window(220, 190, anchor = "nw", window = but4)
    but5_w = can.create_window(220, 230, anchor = "nw", window = but5)
    but6_w = can.create_window(220, 270, anchor = "nw", window = but6)
    but7_w = can.create_window(220, 310, anchor = "nw", window = but7)
    
def disAble():
    can.delete('all')
    can.create_image(0, 0, image = bg, anchor = "nw")
    
def eClear1(e):
    e1.delete(0, END)
    
def eClear2(e):
    e2.delete(0, END)
def pageRankCall():
    disAble()
    global e1
    global e2
    
    def run():
        disAble()           
        try:
            a = float(e1.get())
            p = str(e2.get())
            pageRank(p, a)
        except:
            print('error')
            can.create_text(300, 150, text="Some error accured", font = ("Helvetica", 10), fill = "white")
            btt1 = Button(root, text = "Try again", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
            ew = can.create_window(225, 200, anchor = "nw", window = btt1)
            return
        disAble()
        can.create_text(300, 150, text="output is store in CrawledPapers2.json", font = ("Helvetica", 10), fill = "white")
        bttn1 = Button(root, text = "MENU", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
        ew = can.create_window(225, 200, anchor = "nw", window = bttn1)
            
            
            
            
    e1 = Entry(root, width = 30, fg = "#043249")
    e2 = Entry(root, width = 30, fg = "#043249")
    ew = can.create_window(200, 120, anchor = "nw", window = e1)
    ew = can.create_window(200, 150, anchor = "nw", window = e2)
    e2.insert(0, "insert file path:")
    e1.insert(0, "insert alpha parameter:")
    e1.bind("<Button-1>", eClear1)
    e2.bind("<Button-1>", eClear2)
    butt1 = Button(root, text = "Submit", bg = "#587A98", fg="white", height = 1, width = 20, command = run)
    butt2 = Button(root, text = "MENU", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
    ew = can.create_window(140, 200, anchor = "nw", window = butt2)
    ew = can.create_window(310, 200, anchor = "nw", window = butt1)
    

def CrowlCall():
    
    can.create_text(300, 150, text="Crawling...", font = ("Helvetica", 10), fill = "white")
    root.destroy()
    crawl("start.txt")
    disAble()
    can.create_text(300, 150, text="output is store in CrawledPapers.json", font = ("Helvetica", 10), fill = "white")
    disAble()
    
    
def HITSCall():
    disAble()
    global e1
    global e2
    def run():
        disAble()  
        try:
            a = int(e1.get())
            p = str(e2.get())
            HITS(p, a)
        except:
            print('error')
            can.create_text(300, 150, text="Some error accured", font = ("Helvetica", 10), fill = "white")
            btt1 = Button(root, text = "Try again", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
            can.create_window(225, 200, anchor = "nw", window = btt1)
            return
        disAble()
        bttn1 = Button(root, text = "MENU", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
        can.create_window(225, 350, anchor = "nw", window = bttn1)
            
        file = open("CrawledPapers3.json", "r") 
        doc = json.load(file)
        
        for i in range(len(doc)):
            can.create_text(300, 50 + (i*25), text=doc[str(i+1)], font = ("Helvetica", 10), fill = "white")
        
        
        
    e1 = Entry(root, width = 30, fg = "#043249")
    e2 = Entry(root, width = 30, fg = "#043249")
    ew = can.create_window(200, 120, anchor = "nw", window = e1)
    ew = can.create_window(200, 150, anchor = "nw", window = e2)
    e2.insert(0, "insert file path:")
    e1.insert(0, "insert number of authors:")
    e1.bind("<Button-1>", eClear1)
    e2.bind("<Button-1>", eClear2)
    butt1 = Button(root, text = "Submit", bg = "#587A98", fg="white", height = 1, width = 20, command = run)
    butt2 = Button(root, text = "MENU", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
    ew = can.create_window(140, 200, anchor = "nw", window = butt2)
    ew = can.create_window(310, 200, anchor = "nw", window = butt1)

def CBCall():
    disAble()
    global e2
    def run():
        disAble()
        try:
            a = int(e2.get())
            ContentBased(a, "CrawledPapers.json")
        except:
            print('error')
            can.create_text(300, 150, text="Some error accured", font = ("Helvetica", 10), fill = "white")
            btt1 = Button(root, text = "Try again", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
            ew = can.create_window(225, 200, anchor = "nw", window = btt1)
            return
        disAble()
        
        bttn1 = Button(root, text = "MENU", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
        can.create_window(225, 350, anchor = "nw", window = bttn1)
            
        file = open("CrawledPapers4.json", "r") 
        doc = json.load(file)
        
        for i in range(len(doc)):
            can.create_text(300, 50 + (i*25), text=doc[str(i+1)], font = ("Helvetica", 10), fill = "white")
        
        
     
    e2 = Entry(root, width = 30, fg = "#043249")
    ew = can.create_window(200, 150, anchor = "nw", window = e2)
    e2.insert(0, "insert user profile number:")
    e2.bind("<Button-1>", eClear2)
    butt1 = Button(root, text = "Submit", bg = "#587A98", fg="white", height = 1, width = 20, command = run)
    butt2 = Button(root, text = "MENU", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
    ew = can.create_window(140, 200, anchor = "nw", window = butt2)
    ew = can.create_window(310, 200, anchor = "nw", window = butt1)
    
def CFCall():
    disAble()
    global e1
    global e2
    def run():
        disAble()  
        try:
            a = int(e1.get())
            p = int(e2.get())
            CollaborativeFiltering(p, a, "CrawledPapers.json")
        except:
            print('error')
            can.create_text(300, 150, text="Some error accured", font = ("Helvetica", 10), fill = "white")
            btt1 = Button(root, text = "Try again", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
            ew = can.create_window(225, 200, anchor = "nw", window = btt1)
            return
        disAble()
        
        bttn1 = Button(root, text = "MENU", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
        can.create_window(225, 350, anchor = "nw", window = bttn1)
            
        file = open("CrawledPapers5.json", "r") 
        doc = json.load(file)
        
        for i in range(len(doc)):
            can.create_text(300, 50 + (i*25), text=doc[str(i+1)], font = ("Helvetica", 10), fill = "white")
        
    e1 = Entry(root, width = 30, fg = "#043249")
    e2 = Entry(root, width = 30, fg = "#043249")
    ew = can.create_window(200, 120, anchor = "nw", window = e1)
    ew = can.create_window(200, 150, anchor = "nw", window = e2)
    e2.insert(0, "insert user profile number:")
    e1.insert(0, "insert number of similar users:")
    e1.bind("<Button-1>", eClear1)
    e2.bind("<Button-1>", eClear2)
    butt1 = Button(root, text = "Submit", bg = "#587A98", fg="white", height = 1, width = 20, command = run)
    butt2 = Button(root, text = "MENU", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
    ew = can.create_window(140, 200, anchor = "nw", window = butt2)
    ew = can.create_window(310, 200, anchor = "nw", window = butt1)


def CompleteCall():
    disAble()
    global e1
    global e2
    def run():
        disAble()  
        try:
            a = int(e1.get())
            p = float(e2.get())
            CompleteM(a, p)
        except:
            print('error')
            can.create_text(300, 150, text="Some error accured", font = ("Helvetica", 10), fill = "white")
            btt1 = Button(root, text = "Try again", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
            ew = can.create_window(225, 200, anchor = "nw", window = btt1)
            return
        disAble()
        
        bttn1 = Button(root, text = "MENU", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
        can.create_window(225, 250, anchor = "nw", window = bttn1)
            
        file = open("CrawledPapers6.json", "r") 
        doc = json.load(file)
        
        can.create_text(300, 150, text=doc["error"], font = ("Helvetica", 10), fill = "white")
        
        
        
        
    e1 = Entry(root, width = 30, fg = "#043249")
    e2 = Entry(root, width = 30, fg = "#043249")
    ew = can.create_window(200, 120, anchor = "nw", window = e1)
    ew = can.create_window(200, 150, anchor = "nw", window = e2)
    e2.insert(0, "insert etha parameter:")
    e1.insert(0, "insert number of steps:")
    e1.bind("<Button-1>", eClear1)
    e2.bind("<Button-1>", eClear2)
    butt1 = Button(root, text = "Submit", bg = "#587A98", fg="white", height = 1, width = 20, command = run)
    butt2 = Button(root, text = "MENU", bg = "#753434", fg="white", height = 1, width = 20, command = inable)
    ew = can.create_window(140, 200, anchor = "nw", window = butt2)
    ew = can.create_window(310, 200, anchor = "nw", window = butt1)

    
root = Toplevel()
root.title('MIR3')
root.geometry('600x400')
bg = ImageTk.PhotoImage(Image.open("back.jpg"))
can = Canvas(root, width = 600, height = 400)
can.pack(fill = "both", expand = True)
can.create_image(0, 0, image = bg, anchor = "nw")
but1 = Button(root, text = "Crawl", bg = "#587A98", fg="white", height = 1, width = 20, command = CrowlCall)
but2 = Button(root, text = "Page Rank", bg = "#587A98", fg="white", height = 1, width = 20, command = pageRankCall)
but3 = Button(root, text = "HITS", bg = "#587A98", fg="white", height = 1, width = 20, command = HITSCall)
but4 = Button(root, text = "Content Based", bg = "#587A98", fg="white", height = 1, width = 20, command = CBCall)
but5 = Button(root, text = "Collaborative Filtering", bg = "#587A98", fg="white", height = 1, width = 20, command = CFCall)
but6 = Button(root, text = "Complete Matrix", bg = "#587A98", fg="white", height = 1, width = 20, command = CompleteCall)
but7 = Button(root, text = "Exit", bg = "#753434", fg="white", height = 1, width = 20, command = root.destroy)
inable()




root.mainloop()

