import numpy as np
import collections
import math
import torch



#%%mothod
coden_dict = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,                             # alanine<A>
              'UGU': 1, 'UGC': 1,                                                 # systeine<C>
              'GAU': 2, 'GAC': 2,                                                 # aspartic acid<D>
              'GAA': 3, 'GAG': 3,                                                 # glutamic acid<E>
              'UUU': 4, 'UUC': 4,                                                 # phenylanaline<F>
              'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,                             # glycine<G>
              'CAU': 6, 'CAC': 6,                                                 # histidine<H>
              'AUU': 7, 'AUC': 7, 'AUA': 7,                                       # isoleucine<I>
              'AAA': 8, 'AAG': 8,                                                 # lycine<K>
              'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,         # leucine<L>
              'AUG': 10,                                                          # methionine<M>
              'AAU': 11, 'AAC': 11,                                               # asparagine<N>
              'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,                         # proline<P>
              'CAA': 13, 'CAG': 13,                                               # glutamine<Q>
              'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,   # arginine<R>
              'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,   # serine<S>
              'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,                         # threonine<T>
              'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,                         # valine<V>
              'UGG': 18,                                                          # tryptophan<W>
              'UAU': 19, 'UAC': 19,                                               # tyrosine(Y)
              'UAA': 20, 'UAG': 20, 'UGA': 20,                                    # STOP code
              }
def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**1
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index

def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**2
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index

def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index

def coden(seq,kmer,tris):
    coden_dict = tris
    freq_dict = frequency(seq,kmer,coden_dict)
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i+kmer].replace('T', 'U')]]
        vectors[i][coden_dict[seq[i:i+kmer].replace('T', 'U')]] = value/100
    return vectors

def codenNCP(seq):
    phys_dic = {
        'A': [1, 1, 1],
        'U': [0, 0, 1],
        'C': [0, 1, 0],
        'G': [1, 0, 0]}
    seqLength = len(seq)
    sequence_vector = np.zeros((101, 3))
    for i in range(0, seqLength):
        sequence_vector[i, 0:3] = phys_dic[seq[i].replace('T', 'U')]
    for i in range(seqLength, 101):
        sequence_vector[i, -1] = 1
    return sequence_vector.tolist()


def codenDPCP(seq):
    phys_dic = {
        # Shift Slide Rise Tilt Roll Twist Stacking_energy Enthalpy Entropy Free_energy Hydrophilicity
        'AA': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.04],
        'AU': [-0.06, -1.36, 3.24, 1.1, 7.1, 33, -15.4, -5.7, -15.5, -1.1, 0.14],
        'AC': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.14, ],
        'AG': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.08],
        'UA': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -13.3, -35.5, -2.35, 0.1],
        'UU': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.27],
        'UC': [0.07, -1.39, 3.22, 0, 6.1, 35, -16.9, -14.2, -34.9, -3.42, 0.26],
        'UG': [-0.01, -1.78, 3.32, 0.3, 12.1, 32, -11.1, -12.2, -29.7, -3.26, 0.17],
        'CA': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -10.5, -27.8, -2.11, 0.21],
        'CU': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.52],
        'CC': [-0.01, -1.78, 3.32, 0.3, 8.7, 32, -11.1, -12.2, -29.7, -3.26, 0.49],
        'CG': [0.3, -1.89, 3.3, -0.1, 12.1, 27, -15.6, -8, -19.4, -2.36, 0.35],
        'GA': [-0.02, -1.45, 3.26, -0.2, 10.7, 32, -16, -8.1, -22.6, -1.33, 0.21],
        'GU': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.44],
        'GC': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -10.2, -26.2, -2.35, 0.48],
        'GG': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -7.6, -19.2, -2.11, 0.34]}

    seqLength = len(seq)
    sequence_vector = np.zeros((101, 11))
    k = 2
    for i in range(len(seq)):
        if i < len(seq) - 2:
           sequence_vector[i, 0:11] = phys_dic[seq[i:i + k].replace('T', 'U')]
    return sequence_vector.tolist()

def codenKNF(seq):
    vectors = np.zeros((len(seq), 21))
    for i in range(len(seq)):
        if i < len(seq)-2:
            vectors[i][coden_dict[seq[i:i+3].replace('T', 'U')]] = 1
    return vectors.tolist()

def frequency(seq,kmer,coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i+k]
        kmer_value = coden_dict[kmer.replace('T', 'U')]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict


def Gaussian(x):
    return math.exp(-0.5*(x*x))

#%%Kmer is 101×84
def dealwithdata1(protein):
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    dataX = []
    with open(r'./Datasets/circRNA-RBP/'+protein+'/positive') as f:
        for line in f:
            if '>' not in line:
                kmer1 = coden(line.strip(),1,tris1)
                kmer2 = coden(line.strip(),2,tris2)
                kmer3 = coden(line.strip(),3,tris3)
                Kmer = np.hstack((kmer1,kmer2,kmer3))
                dataX.append(Kmer.tolist())
    with open(r'./Datasets/circRNA-RBP/'+protein+'/negative') as f:
        for line in f:
            if '>' not in line:
                kmer1 = coden(line.strip(),1,tris1)
                kmer2 = coden(line.strip(),2,tris2)
                kmer3 = coden(line.strip(),3,tris3)
                Kmer = np.hstack((kmer1,kmer2,kmer3))
                dataX.append(Kmer.tolist())

    dataX = np.array(dataX)
    return dataX

#%%NCP is 101×3
def dealwithdataNCP(protein):
    dataNCP = []
    with open(r'./Datasets/circRNA-RBP/'+protein+'/positive') as f:
            for line in f:
                if '>' not in line:
                    dataNCP.append(codenNCP(line.strip()))
    with open(r'./Datasets/circRNA-RBP/' + protein + '/negative') as f:
            for line in f:
                if '>' not in line:
                    dataNCP.append(codenNCP(line.strip()))
    dataX = np.array(dataNCP)
    return dataX

#%%DPCP is 101×11
def dealwithdataDPCP(protein):
    dataDPCP = []
    with open(r'./Datasets/circRNA-RBP/'+protein+'/positive') as f:
            for line in f:
                if '>' not in line:
                    dataDPCP.append(codenDPCP(line.strip()))
    with open(r'./Datasets/circRNA-RBP/' + protein + '/negative') as f:
            for line in f:
                if '>' not in line:
                    dataDPCP.append(codenDPCP(line.strip()))
    dataX = np.array(dataDPCP)
    return dataX

#%%KNF is 101×21
def dealwithdataKNF(protein):
    dataXKNF = []
    dataYKNF = []
    with open(r'./Datasets/circRNA-RBP/'+protein+'/positive') as f:
            for line in f:
                if '>' not in line:
                    dataXKNF.append(codenKNF(line.strip()))
                    dataYKNF.append(1)
    with open(r'./Datasets/circRNA-RBP/'+protein+'/negative') as f:
            for line in f:
                if '>' not in line:
                    dataXKNF.append(codenKNF(line.strip()))
                    dataYKNF.append(0)
    dataX = np.array(dataXKNF)
    dataY = np.array(dataYKNF)
    return dataX,dataY

#%%main mothod
def all_data(protein):
    KNF = dealwithdataKNF(protein)
    Kmer = dealwithdata1(protein)
    NCP = dealwithdataNCP(protein)
    DPCP = dealwithdataDPCP(protein)
    knf = KNF[0]
    Y = KNF[1]
    return Kmer,NCP,DPCP,knf,Y





