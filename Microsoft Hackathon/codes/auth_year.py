import gensim, csv, operator

f = open('BingHackathonTrainingData.txt', 'r')
data = f.read().split('\n')[1:-1]

documents = []

auth_yr = {}

for line in data:
    line = line.strip()
    line = line.split('\t')
    authors = line[3].split(';')
    yr = line[2]

    for auth in authors:
        if auth+yr not in auth_yr.keys():
            auth_yr[auth + yr] = 1
        else:
            auth_yr[auth + yr] +=1
    
    
