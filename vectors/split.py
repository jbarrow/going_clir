def split(read, write, languages):
    with open(read) as fp_in:
        count, total = 0, 0
        with open(write, 'w') as fp_out:
            for line in fp_in:
                total += 1
                if line[:2] not in languages: continue
                count += 1
                fp_out.write(line)
        langs = ', '.join(languages)
        
    return total, count
        

if __name__ == '__main__':
    languages = ['en', 'fr']
    read = '../material/twelve.table4.multiCCA.size_512+w_5+it_10.normalized'
    write = './multiCCA_512.en.fr'
    
    total, count = split(read, write, languages)
    
    print(f'Wrote {count} vectors out of {total} (languages: {langs})')