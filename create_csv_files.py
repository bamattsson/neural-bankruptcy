import os


def getCSVFromArff(fileName):
    # Script from http://biggyani.blogspot.co.uk/2014/08/converting-back-and-forth-between-weka.html

    with open(fileName + '.arff', 'r') as fin:
        data = fin.read().splitlines(True)

    i = 0
    cols = []
    for line in data:
        line = line.lower()
        if ('@data' in line):
            i += 1
            break
        else:
            # print line
            i += 1
            if (line.startswith('@attribute')):
                if('{' in line):
                    cols.append(line[11:line.index('{') - 1])
                else:
                    cols.append(line[11:line.index(' ', 11)])

    headers = ",".join(cols)

    with open(fileName + '.csv', 'w') as fout:
        fout.write(headers)
        fout.write('\n')
        fout.writelines(data[i:])


def main():
    main_path = './data/Dane/'
    for year in range(1, 6):
        file_path = os.path.join(main_path, '{}year'.format(year))
        getCSVFromArff(file_path)


if __name__ == '__main__':
    main()
