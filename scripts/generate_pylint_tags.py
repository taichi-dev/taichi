TAGS = {
    'C0121': True,
    'C0415': True,
    'W0611': True,
    'W0202': True,
    'W0621': False,
    'W0622': False,
    'W0401': True,
    'C0209': True,
    'W0404': True,
    'W0612': True,
    'E1101': False,
    'R0402': True,
    'R0201': True,
    'W0235': True,
    'R1705': True,
    'C0200': True,
    'R0205': True,
    'R1732': True,
    'W0101': True,
    'R1710': False,
    'R1703': True,
    'W0108': True,
}

if __name__ == '__main__':
    enabled = [kv[0] for kv in TAGS.items() if kv[1]]
    print(','.join(enabled))
