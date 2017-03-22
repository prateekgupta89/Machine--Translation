import nltk
import A
from collections import defaultdict
from nltk.align import IBMModel1, AlignedSent

class BerkeleyAligner():
    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        alignment = []

        m = len(align_sent.words)
        l = len(align_sent.mots)
        src = align_sent.words
        trg = align_sent.mots

        for j in range(0, m):
            src_word = src[j]
            max_value = self.t[src_word][None] * self.q[0][j+1][m][l]
            idx = -1
            for i in range(0, l):
                trg_word = trg[i]
                value = self.t[src_word][trg_word] * self.q[i + 1][j + 1][m][l]
                if value > max_value:
                    max_value = value
                    idx = i
            if idx != -1:
                alignment.append((j, idx))

        return AlignedSent(align_sent.words, align_sent.mots, alignment)

    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):

        src_vocab = set()
        trg_vocab = set()
        for aligned_sent in aligned_sents:
            src_vocab.update(aligned_sent.words)
            trg_vocab.update(aligned_sent.mots)
        src_vocab.add(None)
        src_vocab.add(None)

        t_ef = IBMModel1(aligned_sents, 5).probabilities

        corpus_fe = []
        for aligned_sent in aligned_sents:
            corpus_fe.append(aligned_sent.invert())

        t_fe = IBMModel1(corpus_fe, 5).probabilities

        q_ef = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
        q_fe = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))

        for aligned_sent in aligned_sents:
            l = len(aligned_sent.mots)
            m = len(aligned_sent.words)

            initial_value = 1.0/(l+1)
            for i in range(0, l+1):
                for j in range(1, m+1):
                    q_ef[i][j][m][l] = initial_value
            
            initial_value = 1.0/(m + 1)
            for i in range(0, m+1):
                for j in range(1, l+1):
                    q_fe[i][j][l][m] = initial_value

        # Start iterations
        for itr in range(0, num_iters):
            count_t_ef = defaultdict(lambda: defaultdict(lambda: 0.0))
            count_t_f = defaultdict(lambda: 0.0)
            count_q_ef = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            count_q_f = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
            count_t_fe = defaultdict(lambda: defaultdict(lambda: 0.0))
            count_t_e = defaultdict(lambda: 0.0)
            count_q_fe = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            count_q_e = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))

            # Expectation step
            for aligned_sent in aligned_sents:
                total_value = defaultdict(lambda: 0.0)
                src = aligned_sent.words
                trg = [None] + aligned_sent.mots
                l = len(trg) - 1
                m = len(src)

                for j in range(1, m+1):
                    src_word = src[j-1]
                    for i in range(0, l+1):
                        trg_word = trg[i]
                        total_value[src_word] += t_ef[src_word][trg_word] * q_ef[i][j][m][l]

                for j in range(1, m+1):
                    src_word = src[j-1]
                    for i in range(0, l+1):
                        trg_word = trg[i]
                        delta = t_ef[src_word][trg_word] * q_ef[i][j][m][l]/total_value[src_word]
                        count_t_ef[src_word][trg_word] += delta
                        count_t_f[trg_word] += delta
                        count_q_ef[i][j][m][l] += delta
                        count_q_f[j][m][l] += delta
               
                total_value = defaultdict(lambda: 0.0)
                src = aligned_sent.mots
                trg = [None] + aligned_sent.words
                l = len(trg) - 1
                m = len(src)

                for j in range(1, m+1):
                    src_word = src[j-1]
                    for i in range(0, l+1):
                        trg_word = trg[i]
                        total_value[src_word] += t_fe[src_word][trg_word] * q_fe[i][j][m][l]

                for j in range(1, m+1):
                    src_word = src[j-1]
                    for i in range(0, l+1):
                        trg_word = trg[i]
                        delta = t_fe[src_word][trg_word] * q_fe[i][j][m][l]/total_value[src_word]
                        count_t_fe[src_word][trg_word] += delta
                        count_t_e[trg_word] += delta
                        count_q_fe[i][j][m][l] += delta
                        count_q_e[j][m][l] += delta

            # Compute average count
            for src_word in count_t_ef:
                for trg_word in count_t_ef[src_word]:
                    avg = (count_t_ef[src_word][trg_word] + count_t_fe[trg_word][src_word])/2.0
                    count_t_ef[src_word][trg_word] = avg
                    count_t_fe[trg_word][src_word] = avg

            for aligned_sent in aligned_sents:
                m = len(aligned_sent.words)
                l = len(aligned_sent.mots)

                for j in range(1, m+1):
                    for i in range(0, l+1):
                        avg = (count_q_ef[i][j][m][l] + count_q_fe[j][i][l][m])/2.0
                        count_q_ef[i][j][m][l] = avg 
                        count_q_fe[j][i][l][m] = avg

            t_ef = defaultdict(lambda: defaultdict(lambda: 0.0))
            q_ef = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            t_fe = defaultdict(lambda: defaultdict(lambda: 0.0))
            q_fe = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))

            # M-step
            for trg_word in trg_vocab:
                for src_word in src_vocab:
                    t_ef[src_word][trg_word] = count_t_ef[src_word][trg_word]/count_t_f[trg_word]

            for src_word in src_vocab:
                for trg_word in trg_vocab:
                    t_fe[trg_word][src_word] = count_t_fe[trg_word][src_word]/count_t_e[src_word]

            # Estimate the new values
            for aligned_sent in aligned_sents:
                l = len(aligned_sent.mots)
                m = len(aligned_sent.words)

                for i in range(0, l+1):
                    for j in range(1, m+1):
                        q_ef[i][j][m][l] = count_q_ef[i][j][m][l]/count_q_f[j][m][l]
                for i in range(0, m+1):
                    for j in range(1, l+1):
                        q_fe[i][j][l][m] = count_q_fe[i][j][l][m]/count_q_e[j][l][m]

        t = t_ef
        q = q_ef
        return t, q

def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
