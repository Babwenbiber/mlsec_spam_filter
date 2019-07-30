import numpy as np
import matplotlib.pyplot as plt
import glob, os, math

def kernel(x, z, d):
    L = set(x.split(" ")).union(set(z.split(" ")))
    return (sum(list(map(lambda w:x.count(w), L))) * sum(list(map(lambda w:z.count(w), L)))) ** d

def norm_kernel(x, z, d):
    return kernel(x,z,d) / math.sqrt(kernel(x, x, d) + kernel(z,z,d))

os.chdir("spam-train")
#get files
files = {"spam":[], "ham":[]}
for file in glob.glob("*.ham.*"):
    files["ham"].append(file)
for file in glob.glob("*.spam.*"):
    files["spam"].append(file)

#get data
spam = []
for file in files["spam"]:
    with open(file, encoding = "ISO-8859-1") as f:
        spam.append(f.read())

d=2
spam_mu = 1.0/len(spam)*sum(list(map(lambda s:norm_kernel(s,s, d), spam)))

ham = []
for file in files["ham"]:
    with open(file, encoding = "ISO-8859-1") as f:
        ham.append(f.read())

d=2
ham_mu = 1.0/len(ham)*sum(list(map(lambda s:norm_kernel(s,s, d), ham)))

print(spam_mu, ham_mu)

os.chdir("/ibr/y-home/y0082315/Desktop/mlsec_ex04")
with open("spam-test/0003.2003-12-18.GP.spam.txt", encoding = "ISO-8859-1") as f:
    test = f.read()
print(test)
print(norm_kernel(test, test, d), norm_kernel(test, test, d))
