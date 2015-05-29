
from struct import unpack
from struct import pack
#import scipy.io.numpyio
# Use the following since fread and fwrite were removed from scipy:
# https://bitbucket.org/lannybroo/numpyio/
import numpyIO
# Read/write a matrix from/to a file binary format.

# fixme: add a version number to the file header. This will help prevent
# accidently reading the wrong version.

def readArray(fileName, type='float'):
    """Read in a matrix of floats or doubles from a data file.
    
    If type = 'float', then use single precision (32 bit) float. This
    is the default.

    If type = 'double', then use double precision (64 bit) float.
    """
    inFile = open(fileName, 'rb')
    dimPacked = inFile.read(4)
    dimensionsTuple = unpack('<i', dimPacked)
    dimensions = dimensionsTuple[0]

    if (dimensions == 1):

        dim1Packed = inFile.read(4)
        dim1Tuple = unpack('<i', dim1Packed)
        dim1 = dim1Tuple[0]

        num = dim1
        if type == 'float':
            flatArray = numpyIO.fread( inFile, num, 'f')
        elif type == 'double':
            flatArray = numpyIO.fread( inFile, num, 'd')
        else:
            raise 'Error: Invalid type.'
        shape = (dim1)
        arrayData =  flatArray.reshape(shape)
    elif (dimensions == 2):

        dim1Packed = inFile.read(4)
        dim1Tuple = unpack('<i', dim1Packed)
        dim1 = dim1Tuple[0]

        dim2Packed = inFile.read(4)
        dim2Tuple = unpack('<i', dim2Packed)
        dim2 = dim2Tuple[0]

        num = dim1*dim2
        if type == 'float':
            flatArray = numpyIO.fread( inFile, num, 'f')
        elif type == 'double':
            flatArray = numpyIO.fread( inFile, num, 'd')
        else:
            raise 'Error: Invalid type.'
        shape = (dim1, dim2)
        arrayData =  flatArray.reshape(shape)
    elif (dimensions == 3):
        dim1Packed = inFile.read(4)
        dim1Tuple = unpack('<i', dim1Packed)
        dim1 = dim1Tuple[0]

        dim2Packed = inFile.read(4)
        dim2Tuple = unpack('<i', dim2Packed)
        dim2 = dim2Tuple[0]

        dim3Packed = inFile.read(4)
        dim3Tuple = unpack('<i', dim3Packed)
        dim3 = dim3Tuple[0]

        num = dim1*dim2*dim3
        if type == 'float':
            flatArray = numpyIO.fread( inFile, num, 'f')
        elif type == 'double':
            flatArray = numpyIO.fread( inFile, num, 'd')
        else:
            raise 'Error: Invalid type.'
        shape = (dim1, dim2, dim3)
        arrayData =  flatArray.reshape(shape)
    elif (dimensions == 4):
        dim1Packed = inFile.read(4)
        dim1Tuple = unpack('<i', dim1Packed)
        dim1 = dim1Tuple[0]

        dim2Packed = inFile.read(4)
        dim2Tuple = unpack('<i', dim2Packed)
        dim2 = dim2Tuple[0]

        dim3Packed = inFile.read(4)
        dim3Tuple = unpack('<i', dim3Packed)
        dim3 = dim3Tuple[0]

        dim4Packed = inFile.read(4)
        dim4Tuple = unpack('<i', dim4Packed)
        dim4 = dim4Tuple[0]

        num = dim1*dim2*dim3*dim4
        if type == 'float':
            flatArray = numpyIO.fread( inFile, num, 'f')
        elif type == 'double':
            flatArray = numpyIO.fread( inFile, num, 'd')
        else:
            raise 'Error: Invalid type.'
        shape = (dim1, dim2, dim3, dim4)
        arrayData =  flatArray.reshape(shape)
    else:
        raise "Error. Only 1 to 4-dimensional arrays (i.e., matrices) are supported!"

    inFile.close
    return arrayData



def writeArray(dataArray, fileName, type='float'):
    """Write the matrix/array of floats or doubles in dataArray to a new file called fileName
    
    If type = 'float', then use single precision (32 bit) float. This
    is the default. If dataArray is of type double, its values will be cast to float
    when written.

    If type = 'double', then use double precision (64 bit) float.
    """
    
    outFile = open(fileName, 'wb')
    # Write the number of dimensions to file:
    numDimensions = dataArray.ndim
    packedDim = pack('<i', numDimensions)
    outFile.write(packedDim)
    # Now write the extent of each dimensions to the file.
    extents = dataArray.shape
    for dim_size in extents:
        packed_size = pack('<i', dim_size)
        outFile.write(packed_size)

    # Now write the actual array data
    numToWrite = dataArray.size
    if type == 'float':
        numpyIO.fwrite(outFile, numToWrite, dataArray, 'f')
    elif type == 'double':
        numpyIO.fwrite(outFile, numToWrite, dataArray, 'd')
    else:
        raise 'Error: Invalid type.'
    outFile.close



