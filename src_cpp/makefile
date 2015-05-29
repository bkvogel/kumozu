
OBJS= MatrixT.o MatrixIO.o Matrix_list.o Utilities.o UnitTests.o LinearLayer.o ConvLayer.o ConvLayer2D.o ConvLayer3D.o Network3DConv3F1.o Network2DConv3F1.o ColumnActivationFunction.o Dropout1D.o Dropout3D.o BoxActivationFunction.o PoolingLayer.o Network.o ExamplesNeuralNet.o ExamplesRecSystem.o NetflixRecSystem.o MinibatchNetworkTrainer.o main.o

OBJS_EXPERIMENTAL= DuplicatedEntryMatrix.o DuplicatedEntryMatrixForW.o MatrixT.o MatrixIO.o NmfSolverSGD.o NmfSolverLeeSeung.o NmfSolverLeeSeungPartialObserved.o Matrix_list.o Utilities.o SolverNegWPosH.o NmfSolverLeeSeungExp1.o UnitTests.o MaxoutLayer.o KmaxLayer.o LinearLayer.o ConvLayer.o ConvLayer2D.o ConvLayer3D.o Network3DConvFull4LExp2.o Network3DConv3F1.o Network2DConv3F1.o Network3DConvFull5LExp1.o ColumnActivationFunction.o Dropout1D.o Dropout3D.o BoxActivationFunction.o PoolingLayer.o Network.o ExamplesNeuralNet.o MinibatchNetworkTrainer.o Examples_experimental.o main_experimental.o
#CC=g++
# Requres at least g++ 4.9
CC=g++-4.9

# Use the following 2 lines for debugging:
#DEBUG = -g -D_GLIBCXX_DEBUG
#OPTIMIZATION = 
# This flag might not work correctly in debug mode:
#OPTIMIZATION = -Og

# Or
# Use the following 2 lines for release mode:
DEBUG =
OPTIMIZATION = -O3 -march=native


#PROFILE = -pg
PROFILE =

# Choose appropriate linker and includes path depending on platform:
##############################################################################3
# Linux:
LINKER_LIB_PATH = -L/home/brian/data/software/openblas_git/OpenBLAS/
# Mac
#LINKER_LIB_PATH = -L/usr/local/opt/openblas/lib -L/usr/local/Cellar/boost/1.57.0/lib
##############################################################################3
# Linux:
INCLUDES_PATH = -I /home/brian/data/software/openblas_git/OpenBLAS/
# Mac:
#INCLUDES_PATH = -I/usr/local/opt/openblas/include -I/usr/local/Cellar/boost/1.57.0/include
##############################################################################3

CCFLAGS = -Wall -fopenmp -std=c++11 -c $(DEBUG) $(OPTIMIZATION)
LINKER_LIBRARIES = -lgomp -lopenblas -lpthread -lgfortran -lboost_filesystem -lboost_system

all: main


main_experimental : $(OBJS_EXPERIMENTAL)
	$(CC) $(LINKER_LIB_PATH) $(OBJS_EXPERIMENTAL) -o main_experimental $(LINKER_LIBRARIES)

main : $(OBJS)
	$(CC) $(PROFILE) $(LINKER_LIB_PATH) $(OBJS) -o main $(LINKER_LIBRARIES)

DuplicatedEntryMatrix.o : DuplicatedEntryMatrix.h DuplicatedEntryMatrix.cpp Utilities.h
	$(CC) $(PROFILE) $(CCFLAGS) DuplicatedEntryMatrix.cpp

DuplicatedEntryMatrixForW.o : DuplicatedEntryMatrixForW.h DuplicatedEntryMatrixForW.cpp Utilities.h
	$(CC) $(PROFILE) $(CCFLAGS) DuplicatedEntryMatrixForW.cpp

MatrixT.o : MatrixT.h MatrixT.cpp
	$(CC) $(PROFILE) $(CCFLAGS) MatrixT.cpp

MatrixIO.o : MatrixIO.h MatrixIO.cpp
	$(CC) $(PROFILE) $(CCFLAGS) MatrixIO.cpp

Utilities.o : Utilities.h Utilities.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Utilities.cpp

UnitTests.o : UnitTests.h UnitTests.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) UnitTests.cpp

SolverNegWPosH.o : SolverNegWPosH.h SolverNegWPosH.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) SolverNegWPosH.cpp

NmfSolverSGD.o : NmfSolverSGD.h NmfSolverSGD.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) NmfSolverSGD.cpp

NmfSolverLeeSeung.o : NmfSolverLeeSeung.h NmfSolverLeeSeung.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) NmfSolverLeeSeung.cpp

NmfSolverLeeSeungExp1.o : NmfSolverLeeSeungExp1.h NmfSolverLeeSeungExp1.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) NmfSolverLeeSeungExp1.cpp

NmfSolverLeeSeungPartialObserved.o : NmfSolverLeeSeungPartialObserved.h NmfSolverLeeSeungPartialObserved.cpp MatrixT.h Utilities.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) NmfSolverLeeSeungPartialObserved.cpp

Matrix_list.o : Matrix_list.h Matrix_list.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Matrix_list.cpp

MaxoutLayer.o : MaxoutLayer.h MaxoutLayer.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) MaxoutLayer.cpp

KmaxLayer.o : KmaxLayer.h KmaxLayer.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) KmaxLayer.cpp

LinearLayer.o : LinearLayer.h LinearLayer.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) LinearLayer.cpp

ConvLayer.o : ConvLayer.h ConvLayer.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ConvLayer.cpp

ConvLayer2D.o : ConvLayer2D.h ConvLayer2D.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ConvLayer2D.cpp

ConvLayer3D.o : ConvLayer3D.h ConvLayer3D.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ConvLayer3D.cpp

Network3DConvFull4LExp2.o : Network3DConvFull4LExp2.h Network3DConvFull4LExp2.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Network3DConvFull4LExp2.cpp

Network3DConv3F1.o : Network3DConv3F1.h Network3DConv3F1.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Network3DConv3F1.cpp

Network2DConv3F1.o : Network2DConv3F1.h Network2DConv3F1.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Network2DConv3F1.cpp

Network3DConvFull5LExp1.o : Network3DConvFull5LExp1.h Network3DConvFull5LExp1.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Network3DConvFull5LExp1.cpp

ColumnActivationFunction.o : ColumnActivationFunction.h ColumnActivationFunction.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ColumnActivationFunction.cpp

Dropout1D.o : Dropout1D.h Dropout1D.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Dropout1D.cpp

Dropout3D.o : Dropout3D.h Dropout3D.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Dropout3D.cpp

BoxActivationFunction.o : BoxActivationFunction.h BoxActivationFunction.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) BoxActivationFunction.cpp

PoolingLayer.o : PoolingLayer.h PoolingLayer.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) PoolingLayer.cpp

Network.o : Network.h Network.cpp MatrixT.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Network.cpp

ExamplesNeuralNet.o : ExamplesNeuralNet.h ExamplesNeuralNet.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ExamplesNeuralNet.cpp

ExamplesRecSystem.o : ExamplesRecSystem.h ExamplesRecSystem.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ExamplesRecSystem.cpp

NetflixRecSystem.o : NetflixRecSystem.h NetflixRecSystem.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) NetflixRecSystem.cpp

Examples_experimental.o : Examples_experimental.h Examples_experimental.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Examples_experimental.cpp

MinibatchNetworkTrainer.o : MinibatchNetworkTrainer.h MinibatchNetworkTrainer.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) MinibatchNetworkTrainer.cpp


main.o : main.cpp MatrixT.h MatrixIO.h Matrix_list.h LinearLayer.h ConvLayer.h ConvLayer2D.h ConvLayer3D.h Network3DConv3F1.h Network2DConv3F1.h Utilities.h ColumnActivationFunction.h Dropout1D.h Dropout3D.h BoxActivationFunction.h PoolingLayer.h Network.h ExamplesNeuralNet.h ExamplesRecSystem.h NetflixRecSystem.h MinibatchNetworkTrainer.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) main.cpp

main_experimental.o : main_experimental.cpp MatrixT.h MatrixIO.h DuplicatedEntryMatrix.h NmfSolverSGD.h NmfSolverLeeSeung.h NmfSolverLeeSeungPartialObserved.h Matrix_list.h MaxoutLayer.h KmaxLayer.h LinearLayer.h ConvLayer.h ConvLayer2D.h ConvLayer3D.h Network2DConvFull.h Network2DConvFull2L.h Network2DConvFull3L.h Network3DConvFull3L.h Network3DConvFull3LMaxout.h Network3DConvFull4LMaxout.h Network3DConvFull4LExp1.h Network3DConvFull5LExp1.h Network3DConvFull4LExp2.h Network3DConv3F1.h Network2DConv3F1.h Utilities.h ColumnActivationFunction.h Dropout1D.h Dropout3D.h BoxActivationFunction.h PoolingLayer.h Network.h ExamplesNeuralNet.h MinibatchNetworkTrainer.h Examples_experimental.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) main_experimental.cpp

clean:
	\rm *.o main
