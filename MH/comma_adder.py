import csv

f = open('./age_groundtruth.csv', 'r', encoding='utf-8')
f_w = open('./age_FGNet.csv', 'w', encoding='utf-8', newline="")
wr = csv.writer(f_w)
rdr = csv.reader(f)
n = []
for line in rdr:
    if len(line) > 2:
        for l in range(len(line)):
            print(">>>>>>>>>>",line)
            i = line[l]
            print(">>> ", i)
            if line.index(i) < len(line)-1:
                if i[0] != '"':
                    i = '"'+i
                if "\n" in i:
                    idx = i.index('\n')
                    i = i[:idx]+'"'+i[idx:]
                if i[-1] != '"':
                    i = '"'+i
                line = i
    n.append(line)
    wr.writerow(line)
print(n)
f.close()
f_w.close()