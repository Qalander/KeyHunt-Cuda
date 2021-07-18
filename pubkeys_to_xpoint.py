import sys


def pubkeys_to_xpoint(filein, fileout):
    with open(filein) as inf, open(fileout, 'wb') as outf:
        count = 0
        skip = 0
        for x in inf.readlines():
            x = x.strip()
            if len(x) != 64:
                if len(x) == 66:
                    x = x[2:]
                elif len(x) == 130:
                    x = x[2:66]
                else:
                    skip += 1
                    print("skipped pubkey:", x)
                    continue
            try:
                outf.write(bytes.fromhex(x))
                count += 1
            except:
                skip += 1
                print("skipped pubkey:", x)

        print('processed :', count, 'pubkeys', '\nskipped   :', skip, 'pubkeys', )


argc = len(sys.argv)
argv = sys.argv

if argc == 1 or argc != 3:
    print('Usage:')
    print('\tpython3 ' + argv[0].replace('\\', '/').split('/')[-1] + ' pubkeys_in.txt xpoints_out.bin')
elif argc == 3:
    pubkeys_to_xpoint(argv[1], argv[2])
