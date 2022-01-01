import ast
import sys
import os

PATH = '/Users/clemenshofstadler/Desktop/'
for filename in os.listdir(PATH):
    if filename.endswith('.qdimacs'):
        with open(os.path.join(PATH, filename), 'r') as f:
            prefix = []
            matrix = []
            for line in f:
                if line.startswith('p'):
                    continue
                elif line.startswith('e'):
                    Q = line.replace('e','')
                    Q = line.replace('0','')
                    Q = line.split()
                    prefix = prefix +
                    TODO
            QBF = f.read().replace(' ','').split(']')
            prefix = ast.literal_eval(QBF[0] + ']')
            truth_value = QBF[-1].replace('\n','')
            matrix = [ast.literal_eval(QBF[i] + ']') for i in   range(1,len(QBF)-1)]
            print(str(prefix).replace(' ',''),str(matrix).replace(' ',''),truth_value)
