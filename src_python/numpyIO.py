''' fread and fwrite using only numpy

Authors: Benyang Tang, btang(noSpam)@pacific.jpl.nasa.gov
         Nickolas Fotopoulos, nvf(noSpam)@mit.edu
         Alan C. Brooks, alancbrooks(noSpam)@gmail.com

========
Functions:
========
a = numpyIO.fread(fid, num, read_type, {mem_type, byteswap})
numpyIO.fwrite(fid, num, myArray, {write_type, byteswap})

========
History:
========
This module, numpyIO, is an implementation in python of the c version of
SciPy's numpyio by Travis Oliphant. The advantage of using numpyIO is the
ease of installation: numpyIO depends only on numpy and does not need scipy.

Initial benchmarking with 10000000 samples indicates that numpyIO is about
5% faster than SciPy's numpyio module.

Only 2 functions were implemented: fread and fwrite. The interfaces of the 
2 functions here are exactly the same as those of numpyio. If you have code
using numpyio, you don't have to change anything to call fread and fwrite, 
except changing numpyio to numpyIO.

See also SciPy's "IO Roadmap": http://projects.scipy.org/scipy/scipy/roadmap

=======
Changelog:
=======
2009-01-03: Read the roadmap and realized that I probably should be using 
            np.fromfile/tofile or np.memmap ... rewrote using to/fromfile
            and now it is often *faster* than SciPy's numpyio.
2009-01-01: Happy with refactored implementation and complete tests (ACB).
2008-12-30: Cleanup and port to numpy 1.3.0 (ACB).
2005-09-02: Added a few new read_types to support Matlab R14 mat-files. Not
            comprehensive. http://projects.scipy.org/scipy/scipy/ticket/14
2003-03-31: Coded and tested.
'''

import numpy as np

def fread(fid, num, read_type, mem_type=None, byteswap=0):
    '''a = numpyIO.fread(fid, num, read_type, {mem_type, byteswap})
     
     fid =       open file pointer object (i.e. from fid = open("filename"))
     num =       number of elements to read of type read_type (-1 for all)
     read_type = a character describing how to interpret bytes on disk:
                    1 byte  => b, B, c, S1
                    2 bytes => h, H
                    4 bytes => f, i, I, l, u4
                    8 bytes => d, F
                   16 bytes => D
OPTIONAL
     mem_type =  a character (PyArray type) describing what kind of
                 PyArray to return in g.   Default = read_type
     byteswap =  0 for no byteswapping or a 1 to byteswap (to handle
                 different endianness).    Default = 0
SEE ALSO
    numpyIO.fwrite, numpy.typecodes, numpy.fromfile'''

    # figure out mem_type
    if not mem_type:
        mem_type = read_type

    # read in
    a = np.fromfile(fid, dtype=read_type, count=num)

    # adjust for byteswap & mem_type
    if byteswap:
        a.byteswap(True)
    if read_type!=mem_type:
        a = a.astype(mem_type)

    return a

def fwrite(fid, num, a, write_type=None, byteswap=0):
    '''numpyIO.fwrite(fid, num, myArray, {write_type, byteswap})
     
     fid =       open file stream
     num =       number of elements to write (-1 for all)
     a =         NumPy array holding the data to write (will be
                 written as if ravel(myarray) was passed)
OPTIONAL
     write_type = character describing how to write data (what datatype
                  to use)                  Default = type of myarray.
                    1 byte  => b, B, c, S1
                    2 bytes => h, H
                    4 bytes => f, i, I, l, u4
                    8 bytes => d, F
                   16 bytes => D
     byteswap =   0 or 1 to determine if byteswapping occurs on write.
                  Default = 0.
SEE ALSO
    numpyIO.fread, numpy.typecodes, a.tofile'''

    # figure out inputs
    mem_type = a.dtype.char
    if not write_type:
        write_type = mem_type
    if num==-1:
        num = a.size

    # if not all elements are written
    if num!=a.size:
        a = a.ravel()[:num]

    # adjust for byteswap & write_type
    if mem_type!=write_type:
        a = a.astype(write_type)
    if byteswap:
        a = a.byteswap()
    
    # write
    a.tofile(fid)
    
    
def test():
    '''Run all unit tests (recommend running with 'nosetests -v')'''
    import test_numpyIO
    test_numpyIO.runAll()
        
if __name__=='__main__':
    test()
