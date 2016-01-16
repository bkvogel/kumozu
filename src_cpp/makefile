
OBJS= Matrix.o MatrixIO.o Matrix_list.o Utilities.o UnitTests.o LinearLayer.o BatchNormalization1D.o BatchNormalization3D.o ImageToColumnLayer.o Layer.o ConvLayer3D.o ColumnActivationFunction.o Dropout1D.o Dropout3D.o BoxActivationFunction.o PoolingLayer.o SequentialNetwork.o ExamplesNeuralNet.o ExamplesRecSystem.o NetflixRecSystem.o MinibatchTrainer.o Updater.o PlotUtilities.o CostFunction.o MSECostFunction.o CrossEntropyCostFunction.o Accumulator.o ExamplesPlot.o ExamplesMatrix.o main.o

OBJS_EXPERIMENTAL= Matrix.o MatrixIO.o Matrix_list.o Utilities.o UnitTests.o LinearLayer.o BatchNormalization1D.o BatchNormalization3D.o ImageToColumnLayer.o Layer.o ConvLayer3D.o ColumnActivationFunction.o Dropout1D.o Dropout3D.o BoxActivationFunction.o PoolingLayer.o SequentialNetwork.o ExamplesNeuralNet.o ExamplesRecSystem.o NetflixRecSystem.o MinibatchTrainer.o Updater.o PlotUtilities.o CostFunction.o MSECostFunction.o CrossEntropyCostFunction.o Accumulator.o main.o
#CC=g++
# Requres at least g++ 4.9
CC=g++-4.9

##########################################################################
# Use the following 2 lines for debugging:
#DEBUG = -g -D_GLIBCXX_DEBUG -DKUMOZU_DEBUG
#OPTIMIZATION = 

##### Or

# Use the following 2 lines for release mode:
DEBUG =
OPTIMIZATION = -O3 -march=native
##########################################################################

# No longer used
PROFILE =

# Choose appropriate linker and includes path depending on platform:
##############################################################################3
# Ubuntu Linux:
LINKER_LIB_PATH = -L /opt/OpenBLAS/lib

##############################################################################3
# Ubuntu Linux:
INCLUDES_PATH = -I /opt/OpenBLAS/include

##############################################################################3

CCFLAGS = -Wall -fopenmp -std=c++11 -c $(DEBUG) $(OPTIMIZATION)
# This will complain about conversions from int to float etc...
#CCFLAGS = -Wfloat-conversion -Wall -fopenmp -std=c++11 -c $(DEBUG) $(OPTIMIZATION)

#LINKER_LIBRARIES = -lgomp -lopenblas -lpthread -lgfortran -lboost_filesystem -lboost_system
# Note: -lboost_iostreams is needed by Gnuplot library: gnuplot-iostream.h
LINKER_LIBRARIES = -lgomp -lopenblas -lpthread -lboost_iostreams -lboost_filesystem -lboost_system

all: main


main_experimental : $(OBJS_EXPERIMENTAL)
	$(CC) $(LINKER_LIB_PATH) $(OBJS_EXPERIMENTAL) -o main $(LINKER_LIBRARIES)

main : $(OBJS)
	$(CC) $(PROFILE) $(LINKER_LIB_PATH) $(OBJS) -o main $(LINKER_LIBRARIES)

Matrix.o : Matrix.h Matrix.cpp
	$(CC) $(PROFILE) $(CCFLAGS) Matrix.cpp

MatrixIO.o : MatrixIO.h MatrixIO.cpp
	$(CC) $(PROFILE) $(CCFLAGS) MatrixIO.cpp

Utilities.o : Utilities.h Utilities.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Utilities.cpp

UnitTests.o : UnitTests.h UnitTests.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) UnitTests.cpp

Matrix_list.o : Matrix_list.h Matrix_list.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Matrix_list.cpp

LinearLayer.o : LinearLayer.h LinearLayer.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) LinearLayer.cpp

BatchNormalization1D.o : BatchNormalization1D.h BatchNormalization1D.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) BatchNormalization1D.cpp

BatchNormalization3D.o : BatchNormalization3D.h BatchNormalization3D.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) BatchNormalization3D.cpp

ImageToColumnLayer.o : ImageToColumnLayer.h ImageToColumnLayer.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ImageToColumnLayer.cpp

Layer.o : Layer.h Layer.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Layer.cpp

ConvLayer3D.o : ConvLayer3D.h ConvLayer3D.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ConvLayer3D.cpp

ColumnActivationFunction.o : ColumnActivationFunction.h ColumnActivationFunction.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ColumnActivationFunction.cpp

Dropout1D.o : Dropout1D.h Dropout1D.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Dropout1D.cpp

Dropout3D.o : Dropout3D.h Dropout3D.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Dropout3D.cpp

BoxActivationFunction.o : BoxActivationFunction.h BoxActivationFunction.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) BoxActivationFunction.cpp

PoolingLayer.o : PoolingLayer.h PoolingLayer.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) PoolingLayer.cpp

SequentialNetwork.o : SequentialNetwork.h SequentialNetwork.cpp Matrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) SequentialNetwork.cpp

ExamplesNeuralNet.o : ExamplesNeuralNet.h ExamplesNeuralNet.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ExamplesNeuralNet.cpp

ExamplesRecSystem.o : ExamplesRecSystem.h ExamplesRecSystem.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ExamplesRecSystem.cpp

NetflixRecSystem.o : NetflixRecSystem.h NetflixRecSystem.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) NetflixRecSystem.cpp

MinibatchTrainer.o : MinibatchTrainer.h MinibatchTrainer.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) MinibatchTrainer.cpp

Updater.o : Updater.h Updater.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Updater.cpp

PlotUtilities.o : PlotUtilities.h PlotUtilities.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) PlotUtilities.cpp

CostFunction.o : CostFunction.h CostFunction.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) CostFunction.cpp

MSECostFunction.o : MSECostFunction.h MSECostFunction.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) MSECostFunction.cpp

CrossEntropyCostFunction.o : CrossEntropyCostFunction.h CrossEntropyCostFunction.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) CrossEntropyCostFunction.cpp

Accumulator.o : Accumulator.h Accumulator.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) Accumulator.cpp

ExamplesPlot.o : ExamplesPlot.h ExamplesPlot.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ExamplesPlot.cpp

ExamplesMatrix.o : ExamplesMatrix.h ExamplesMatrix.cpp
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) ExamplesMatrix.cpp

# 
main.o : main.cpp Matrix.h MatrixIO.h Matrix_list.h LinearLayer.h BatchNormalization1D.h BatchNormalization3D.h ImageToColumnLayer.h Layer.h ConvLayer3D.h Utilities.h ColumnActivationFunction.h Dropout1D.h Dropout3D.h BoxActivationFunction.h PoolingLayer.h SequentialNetwork.h ExamplesNeuralNet.h ExamplesRecSystem.h NetflixRecSystem.h MinibatchTrainer.h Updater.h PlotUtilities.h CostFunction.h MSECostFunction.h CrossEntropyCostFunction.h Accumulator.h ExamplesPlot.h ExamplesMatrix.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) main.cpp

main_experimental.o : main_experimental.cpp Matrix.h MatrixIO.h Matrix_list.h LinearLayer.h BatchNormalization1D.h BatchNormalization3D.h ImageToColumnLayer.h Layer.h ConvLayer3D.h Utilities.h ColumnActivationFunction.h Dropout1D.h Dropout3D.h BoxActivationFunction.h PoolingLayer.h SequentialNetwork.h ExamplesNeuralNet.h ExamplesRecSystem.h NetflixRecSystem.h MinibatchTrainer.h Updater.h PlotUtilities.h CostFunction.h MSECostFunction.h CrossEntropyCostFunction.h Accumulator.h
	$(CC) $(PROFILE) $(CCFLAGS) $(INCLUDES_PATH) main_experimental.cpp

clean:
	\rm *.o main
