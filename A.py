import nltk
from nltk.corpus import comtrans
from nltk.align import IBMModel1, IBMModel2

# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):
    ibm1 = IBMModel1(aligned_sents, 10)
    return ibm1

# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    ibm2 = IBMModel2(aligned_sents, 10)
    return ibm2

# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    total_aer = 0.0

    for i in range(0,n):
        x = aligned_sents[i]
        aligned_sent = model.align(x)
        total_aer = total_aer + float(aligned_sent.alignment_error_rate(x))

    avg_aer = total_aer/float(n)
    return avg_aer

# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    f = open(file_name, 'w')

    num_sentences = 20

    for i in range(0, num_sentences):
        x = aligned_sents[i]
        aligned_sent = model.align(x)
        f.write(str(aligned_sent.words))
        f.write('\n')
        f.write(str(aligned_sent.mots))
        f.write('\n')
        f.write(str(aligned_sent.alignment))
        f.write('\n')
        f.write('\n')

    f.close()

def main(aligned_sents):
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
