const_kather19 = {
    'ADI': ('ADI', 0), 'BACK': ('BACK', 1), 'DEB': ('DEB', 2), 'LYM': ('LYM', 3), 'MUC': ('MUC', 4),
    'MUS': ('MUS', 5), 'NORM': ('NORM', 6), 'STR': ('STR', 7), 'TUM': ('TUM', 8)
}

const_crctp = {
    'Benign': ('NORM', 0), 'Complex Stroma': ('CSTR', 1), 'Debris': ('DEB', 2),
    'Inflammatory': ('LYM', 3), 'Muscle': ('MUS', 4), 'Stroma': ('STR', 5), 'Tumor': ('TUM', 6)
}

const_crctp_to_kather19 = {
    'Benign': ('NORM', 6), 'Complex Stroma': ('CSTR', 9), 'Debris': ('DEB', 2),
    'Inflammatory': ('LYM', 3), 'Muscle': ('MUS', 5), 'Stroma': ('STR', 7), 'Tumor': ('TUM', 8)
}

const_crctp_cstr_to_kather19 = {
    'Complex Stroma': ('CSTR', 9)
}

const_unknown = {
    '*': ('Unknown', -1),
}