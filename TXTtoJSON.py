# coding=utf8

from email import errors
import json
from collections import OrderedDict

lines = []
f = open('C:\\Users\\helen\\Desktop\\연구과제 관련\\Parser\\Depencency_parsing(1)\\3d\\3d3.txt', "r", encoding="cp949")
for line in f:
    if not line.isspace():
        lines.append(line)
f.close()

a = open('C:/Users/choi4/source/repos/AFTERTEXTDATA/3d/3d3.txt', "w", encoding='cp949')
for line in lines:
    a.write(line)
a.close

a = open('C:/Users/choi4/source/repos/AFTERTEXTDATA/3d/3d3.txt', "rt", encoding="cp949")
lines = a.readline()
for line in lines:
    item = line.split("\n")
    applicant = item[item.index("【출원인】")+2]
    applicant = applicant.decode('cp949', errors='ignore')
    agent = item[item.index("【대리인】")+2]
    agent = agent.decode('cp949', errors='ignore')
    kor_name = line[item.index("【발명(고안)의 국문명칭】")+1 : item.index("【발명(고안)의 영문명칭】")-1]
    kor_name = kor_name.decode('cp949', errors='ignore')
    eng_name = line[item.index("【발명(고안)의 영문명칭】")+1 : item.index("【발명(고안)자】")-1]
    eng_name = eng_name.decode('cp949', errors='ignore')
    invent = line[item.index("【발명(고안)자】")+2 : item.index("【특허고객번호】")-1]
    invent = invent.decode('cp949', errors='ignore')
    summary = line[item.index("【요약서】")+1 : item.index("【대표도】")-1]
    summary = summary.decode('cp949', errors='ignore')

file_data = OrderedDict()

file_data["출원인"] = {'명칭':'%s' % (applicant)}
file_data["대리인"] = {'성명':'%s' % (agent)}
file_data["발명(고안)의 국문명칭"] = '%s' % (kor_name)
file_data["발명(고안)의 영문명칭"] = '%s' % (eng_name)
file_data["발명(고안)자"] = {'성명':'%s' % (invent)}
file_data["요약"] = '%s' % (summary)

print(json.dumps(file_data, ensure_ascii=False, indent='\t'))

with open('C:/Users/choi4/source/repos/JSONDATA/3d/3d3.json', 'wt', encoding="cp949") as make_file:
    json.dump(file_data, make_file, ensure_ascii=False, indent='\t')

a.close