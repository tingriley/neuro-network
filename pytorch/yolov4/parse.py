def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')             # store the lines in a list
    lines = [x for x in lines if len(x) > 0]    # remove empty lines
    lines = [x for x in lines if x[0] != '#']   # remove comments
    lines = [x.rstrip().lstrip() for x in lines] 

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block)!=0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

b = parse_cfg('./cfg/yolov4.cfg')
print(b[0])
print("")
print(b[1])


