# cython: language_level=3
# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
cimport numpy as np
cimport cython
from libc.string cimport strlen
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from .matrix import BLOSUM62


# Container for alignment result
AlignmentResult = namedtuple(
    'AlignmentResult',
    ['seq1', 'seq2', 'start1', 'start2',
     'end1', 'end2', 'n_gaps1', 'n_gaps2',
     'n_mismatches', 'score'])
cdef struct aln_res:
    char* seq1
    char* seq2
    int start1, start2, end1, end2, n_gaps1, n_gaps2, n_mismatches
    double score

cdef:
    # Directions for traceback
    int NONE = 0, LEFT = 1, UP = 2, DIAG = 3
    # Character to represent gaps
    char GAP_CHAR = b'-'

# Supported methods
METHODS = {
    "global": 0, "local": 1, "glocal": 2, "global_cfe": 3,
}

ctypedef np.int_t DTYPE_INT
ctypedef np.uint_t DTYPE_UINT
ctypedef np.float32_t DTYPE_FLOAT


cdef inline DTYPE_FLOAT max3(DTYPE_FLOAT a, DTYPE_FLOAT b, DTYPE_FLOAT c):
    if c > b:
        return c if c > a else a
    return b if b > a else a


cdef inline DTYPE_FLOAT max2(DTYPE_FLOAT a, DTYPE_FLOAT b):
    return b if b > a else a


cdef DTYPE_FLOAT[:, :] make_cmatrix(dict pymatrix):
    cdef:
        DTYPE_INT size = sorted([ord(c) for c in pymatrix.keys()]).pop() + 1
        DTYPE_FLOAT score
        np.int8_t c1, c2
        np.ndarray[DTYPE_FLOAT, ndim=2] cmatrix = np.zeros((size, size),
                                                           dtype=np.float32)

    for char1 in pymatrix.keys():
        for char2, score in pymatrix[char1].items():
            c1 = ord(char1)
            c2 = ord(char2)
            cmatrix[c1, c2] = score

    return cmatrix


cdef list caligner(char* seqi, char* seqj, int method,
    DTYPE_FLOAT gap_open, DTYPE_FLOAT gap_extend, DTYPE_FLOAT gap_double,
    DTYPE_FLOAT[:, :] matrix, max_hits, bint flip):

    cdef:
        int max_i = strlen(seqi)
        int max_j = strlen(seqj)
        np.ndarray[DTYPE_FLOAT, ndim=2] F = np.zeros((max_i + 1, max_j + 1), dtype=np.float32)
        DTYPE_FLOAT[:, :] Fview = F
        DTYPE_FLOAT[:, :] I = np.ndarray((max_i + 1, max_j + 1), dtype=np.float32)
        DTYPE_FLOAT[:, :] J = np.ndarray((max_i + 1, max_j + 1), dtype=np.float32)
        DTYPE_UINT[:, :] pointer = np.zeros((max_i + 1, max_j + 1), dtype=np.uint)  # NONE
        int i, j, ci, cj
        DTYPE_FLOAT diag_score, left_score, up_score, max_score
        int NONE = 0, LEFT = 1, UP = 2, DIAG = 3
        # methods: global: 0, local: 1, glocal: 2, global_cfe: 3
        list ij_pairs = [], results = []
        int p, end_i, end_j, n_gaps_i, n_gaps_j, n_mmatch, seqlen, idx, aln_counter
        unsigned char* align_i
        unsigned char* align_j
        DTYPE_FLOAT score

    I[:, :] = -np.inf
    J[:, :] = -np.inf

    if method == 0:
        pointer[0, 1:] = LEFT
        pointer[1:, 0] = UP
        F[0, 1:] = gap_open + gap_extend * \
            np.arange(0, max_j, dtype=np.float32)
        F[1:, 0] = gap_open + gap_extend * \
            np.arange(0, max_i, dtype=np.float32)
    elif method == 3:
        pointer[0, 1:] = LEFT
        pointer[1:, 0] = UP
    elif method == 2:
        pointer[0, 1:] = LEFT
        F[0, 1:] = gap_open + gap_extend * \
            np.arange(0, max_j, dtype=np.float32)

    with cython.boundscheck(False), cython.wraparound(False):
        for i in range(1, max_i + 1):
            ci = seqi[i - 1]
            for j in range(1, max_j + 1):
                cj = seqj[j - 1]
                # I
                I[i, j] = max3(
                            F[i, j - 1] + gap_open,
                            I[i, j - 1] + gap_extend,
                            J[i, j - 1] + gap_double)
                # J
                J[i, j] = max3(
                            F[i - 1, j] + gap_open,
                            J[i - 1, j] + gap_extend,
                            I[i - 1, j] + gap_double)
                # F
                diag_score = Fview[i - 1, j - 1] + matrix[cj][ci]
                left_score = I[i, j]
                up_score = J[i, j]
                max_score = max3(diag_score, up_score, left_score)

                Fview[i, j] = max2(0, max_score) if method == 1 else max_score

                if method == 1:
                    if Fview[i, j] == 0:
                        pass  # point[i,j] = NONE
                    elif max_score == diag_score:
                        pointer[i, j] = DIAG
                    elif max_score == up_score:
                        pointer[i, j] = UP
                    elif max_score == left_score:
                        pointer[i, j] = LEFT
                elif method == 2:
                    # In a semi-global alignment we want to consume as much as
                    # possible of the longer sequence.
                    if max_score == up_score:
                        pointer[i, j] = UP
                    elif max_score == diag_score:
                        pointer[i, j] = DIAG
                    elif max_score == left_score:
                        pointer[i, j] = LEFT
                else:
                    # global
                    if max_score == up_score:
                        pointer[i, j] = UP
                    elif max_score == left_score:
                        pointer[i, j] = LEFT
                    else:
                        pointer[i, j] = DIAG

    cdef:
        int row_max, col_max, ncol_idces, nrow_idces, cid, rid
        long[:, :] col_idces, row_idces

    if method == 1:
        maxv_indices = np.argwhere(F == F.max())[:max_hits]
        for index in maxv_indices:
            ij_pairs.append(index)
    elif method == 2:
        # max in last col
        maxi_indices = np.argwhere(F[:, -1] == F[:, -1].max())\
            .flatten()[:max_hits]
        for i in maxi_indices:
            ij_pairs.append((i, max_j))
    elif method == 3:
        # from i,j to max(max(last row), max(last col)) for free
        row_max = F[-1].max()
        col_max = F[:, -1].max()
        # expecting max to exist on either last column or last row
        if row_max > col_max:
            col_idces = np.argwhere(F[-1] == row_max).flatten()[:max_hits]
            for cid in col_idces:
                ij_pairs.append((max_i, cid))
        elif row_max < col_max:
            row_idces = np.argwhere(F[:, -1] == col_max).flatten()[:max_hits]
            for rid in row_idces:
                ij_pairs.append((rid, max_j))
        # special case: max is on last row, last col
        elif row_max == col_max == F[max_i, max_j]:
            # check if max score also exist on other cells in last row
            # or last col. we expect only one of the case.
            col_idces = np.argwhere(F[-1] == row_max).flatten()
            row_idces = np.argwhere(F[:, -1] == col_max).flatten()
            ncol_idces = len(col_idces)
            nrow_idces = len(row_idces)

            # tiebreaker between row/col is whichever has more max scores
            if ncol_idces > nrow_idces:
                for cid in col_idces[:max_hits]:
                    ij_pairs.append((max_i, cid))
            elif ncol_idces < nrow_idces:
                for rid in row_idces[:max_hits]:
                    ij_pairs.append((rid, max_j))
            elif ncol_idces == nrow_idces == 1:
                ij_pairs.append((max_i, max_j))
            else:
                raise RuntimeError('Unexpected multiple maximum global_cfe'
                                   ' scores.')
        else:
            raise RuntimeError('Unexpected global_cfe scenario.')
    else:
        # method must be global at this point
        ij_pairs.append((max_i, max_j))

    for i, j in ij_pairs:
        score = F[i, j]
        p = pointer[i, j]
        # mimic Python's coord system
        if method == 0 or method == 3:
            end_i, end_j = max_i, max_j
        else:
            end_i, end_j = i, j
        n_gaps_i, n_gaps_j, n_mmatch = 0, 0, 0
        aln_counter = 0
        seqlen = max_i + max_j
        align_i = <unsigned char *>PyMem_Malloc(seqlen * sizeof(unsigned char))
        align_j = <unsigned char *>PyMem_Malloc(seqlen * sizeof(unsigned char))

        # special case for global_cfe ~ one cell may contain multiple pointer
        # directions
        if method == 3:
            if i < max_i:
                n_gaps_j += 1
                for idx in range(max_i - i):
                    align_j[idx] = GAP_CHAR
                    align_i[idx] = seqi[-1 * (idx + 1)]
            elif j < max_j:
                n_gaps_i += 1
                for idx in range(max_j - j):
                    align_i[idx] = GAP_CHAR
                    align_j[idx] = seqj[-1 * (idx + 1)]

        while p != NONE:
            if p == DIAG:
                i -= 1
                j -= 1
                if seqi[i] != seqj[j]:
                    n_mmatch += 1
                align_j[aln_counter] = seqj[j]
                align_i[aln_counter] = seqi[i]
            elif p == LEFT:
                j -= 1
                align_j[aln_counter] = seqj[j]
                if align_i[aln_counter - 1] != GAP_CHAR or aln_counter == 0:
                    n_gaps_i += 1
                align_i[aln_counter] = GAP_CHAR
            elif p == UP:
                i -= 1
                align_i[aln_counter] = seqi[i]
                if align_j[aln_counter - 1] != GAP_CHAR or aln_counter == 0:
                    n_gaps_j += 1
                align_j[aln_counter] = GAP_CHAR
            else:
                raise Exception('wtf!')
            p = pointer[i, j]
            aln_counter += 1

        alns_i = bytes(align_i[:aln_counter][::-1])
        alns_j = bytes(align_j[:aln_counter][::-1])

        PyMem_Free(align_i)
        PyMem_Free(align_j)

        aln = (AlignmentResult(alns_i, alns_j, i, j, end_i, end_j,
                            n_gaps_i, n_gaps_j, n_mmatch, score)
            if flip else
            AlignmentResult(alns_j, alns_i, j, i, end_j, end_i,
                            n_gaps_j, n_gaps_i, n_mmatch, score))

        results.append(aln)

    return results


def aligner(seqj, seqi, method='global', gap_open=-7, gap_extend=-7,
            gap_double=-7, matrix=BLOSUM62, max_hits=1):
    '''Calculates the alignment of two sequences.

    The supported 'methods' are:
        * 'global' for a global Needleman-Wunsh algorithm
        * 'local' for a local Smith-Waterman alignment
        * 'global_cfe' for a global alignment with cost-free ends
        * 'glocal' for an alignment which is 'global' only with respect to
          the shorter sequence (also known as a 'semi-global' alignment)

    Returns the aligned (sub)sequences as character arrays.

    Gotoh, O. (1982). J. Mol. Biol. 162, 705-708.
    Needleman, S. & Wunsch, C. (1970). J. Mol. Biol. 48(3), 443-53.
    Smith, T.F. & Waterman M.S. (1981). J. Mol. Biol. 147, 195-197.

    Arguments:

        - seqj (``sequence``) First aligned iterable object of symbols.
        - seqi (``sequence``) Second aligned iterable object of symbols.
        - method (``str``) Type of alignment: 'global', 'global_cfe', 'local',
          'glocal'.
        - gap_open (``float``) The gap-opening cost.
        - gap_extend (``float``) The cost of extending an open gap.
        - gap_double (``float``) The gap-opening cost if a gap is already open
          in the other sequence.
        - matrix (``dict``) A score matrix dictionary.
        - max_hits (``int``) The maximum number of results to return in
          case multiple alignments with the same score are found. If set to 1,
          a single ``AlignmentResult`` object is returned. If set to values
          larger than 1, a list containing ``AlignmentResult`` objects are
          returned. If set to `None`, all alignments with the maximum score
          are returned.
    '''
    assert max_hits is None or max_hits > 0
    max_j = len(seqj)
    max_i = len(seqi)

    if max_j > max_i:
        flip = 1
        seqi, seqj = seqj, seqi
        max_i, max_j = max_j, max_i
    else:
        flip = 0

    seq1 = seqi if isinstance(seqi, bytes) else bytes(seqi, 'ascii')
    seq2 = seqj if isinstance(seqj, bytes) else bytes(seqj, 'ascii')

    return caligner(seq1, seq2,
                    METHODS[method], gap_open, gap_extend, gap_double,
                    make_cmatrix(matrix), max_hits, flip)
