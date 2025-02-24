import os


def write_motif_pattern_to_jaspar(motif_pattern, filename):
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    pfm = {aa: [0] * len(motif_pattern) for aa in alphabet}
    for i, char in enumerate(motif_pattern):
        if char == '*':
            for aa in alphabet:
                pfm[aa][i] = 1
        else:
            for aa in alphabet:
                if aa == char:
                    pfm[aa][i] = 20
                else:
                    pfm[aa][i] = 0
    with open(filename, 'w') as f:
        f.write(f'>{motif_pattern}\n')
        for aa in alphabet:
            counts = ' '.join(str(x) for x in pfm[aa])
            f.write(f'{aa} [ {counts} ]\n')


def main():
    motif_patterns_dir = '../motif_patterns/'
    motifs = ["I*VN", "W*C*Y*", "E*N*D*", "AADD", "*A*I*A", "GH*L*", "Y*WG", "Q*Y*T*", "A*I*H", "MA*G"]

    os.makedirs(motif_patterns_dir, exist_ok=True)
    for i, motif in enumerate(motifs):
        write_motif_pattern_to_jaspar(motif, f'{motif_patterns_dir}/motif{i+1}.jaspar')


if __name__ == '__main__':
    main()
