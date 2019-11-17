import numpy as np
path = r'C:\Users\user\Desktop\\'
traces = np.load(path + 'traces.npy')
text   = np.load(path + 'textin.npy')
HW = [bin(n).count("1") for n in range(0,256)]
SBOXES = {"Gost28147_tc26_ParamZ": (
        (12, 4, 6, 2, 10, 5, 11, 9, 14, 8, 13, 7, 0, 3, 15, 1),
        (6, 8, 2, 3, 9, 10, 5, 12, 1, 14, 4, 7, 11, 13, 0, 15),
        (11, 3, 5, 8, 2, 15, 10, 13, 14, 1, 7, 4, 12, 9, 6, 0),
        (12, 8, 2, 1, 13, 4, 15, 6, 7, 0, 10, 5, 3, 14, 9, 11),
        (7, 15, 5, 10, 8, 1, 6, 13, 0, 9, 3, 14, 11, 4, 2, 12),
        (5, 13, 15, 6, 9, 2, 12, 10, 11, 7, 8, 1, 4, 3, 14, 0),
        (8, 14, 2, 5, 6, 9, 1, 12, 15, 4, 11, 0, 13, 10, 3, 7),
        (1, 7, 14, 13, 0, 5, 8, 3, 4, 15, 10, 6, 9, 12, 11, 2),
    )}

def _K(s, _in):
    """ S-box substitution
    :param s: S-box
    :param _in: 32-bit word
    :returns: substituted 32-bit word
    """
    return (
        (s[0][(_in >> 0) & 0x0F] << 0) +
        (s[1][(_in >> 4) & 0x0F] << 4) +
        (s[2][(_in >> 8) & 0x0F] << 8) +
        (s[3][(_in >> 12) & 0x0F] << 12) +
        (s[4][(_in >> 16) & 0x0F] << 16) +
        (s[5][(_in >> 20) & 0x0F] << 20) +
        (s[6][(_in >> 24) & 0x0F] << 24) +
        (s[7][(_in >> 28) & 0x0F] << 28)
    )
def block2ns(data):
    """ Convert block to N1 and N2 integers
    """
    data = bytearray(data)
    return (
        data[7] | data[6] << 8 | data[5] << 16 | data[4] << 24,
        data[3] | data[2] << 8 | data[1] << 16 | data[0] << 24,
    )
def addmod(x, y, mod=2 ** 32):
    """ Modulo adding of two integers
    """
    r = x + int(y)
    return r if r < mod else r - mod
def _shift11(x):
    """ 11-bit cyclic shift
    """
    return ((x << 11) & (2 ** 32 - 1)) | ((x >> (32 - 11)) & (2 ** 32 - 1))
def leak(sbox, key, data, nbyte):
    s = SBOXES[sbox]
    _in = addmod(data, key)
    sbox_leak = _K(s, _in);
    return (sbox_leak >> (8 * nbyte)) & 0xFF
numtraces = len(traces)
numpoint = np.shape(traces)[1]
cpaoutput = [0]*256
maxcpa = [0]*256

for kguess in range(0,  256):
	#Initialize arrays & variables to zero
	sumnum = np.zeros(numpoint)
	sumden1 = np.zeros(numpoint)
	sumden2 = np.zeros(numpoint)
	hyp = np.zeros(numtraces)
	for tnum in range(numtraces):
	    hyp[tnum] = HW[leak("Gost28147_tc26_ParamZ",  kguess, block2ns(text[tnum])[0], 0)]
	#Mean of hypothesis
	meanh = np.mean(hyp, dtype=np.float64)
	#Mean of all points in trace
	meant = np.mean(traces, axis=0, dtype=np.float64)
	#For each trace, do the following
	for tnum in range(0, numtraces):
		hdiff = (hyp[tnum] - meanh)
		tdiff = traces[tnum,:] - meant
		sumnum = sumnum + (hdiff*tdiff)
		sumden1 = sumden1 + hdiff*hdiff
		sumden2 = sumden2 + tdiff*tdiff
	cpaoutput[kguess] = sumnum / np.sqrt( sumden1 * sumden2 )
	maxcpa[kguess] = max(abs(cpaoutput[kguess]))
bestguess = np.argmax(maxcpa)

    
print ("Best Key Guess: ")
print ("%02x "%bestguess, end = '\n')
