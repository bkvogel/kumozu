#ifndef _VARIABLE_H
#define _VARIABLE_H
/*
 * Copyright (c) 2005-2015, Brian K. Vogel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 *
 */
//#include <string>
//#include <vector>

//#include <iostream>
//#include <sstream>
//#include <string>
//#include <random>
//#include <ctime>
//#include "Assertions.h"
#include "Matrix.h"

namespace kumozu {



/**
 * A container of two N-dimensional matrices of Type T having identical extents.
 *
 * A Variable contains two matrices of the same extents: the "data" matrix and the
 * "grad" matrix. A Variable is intended to be used as a component of a deep learning
 * framework where it is typically required to maintain two such matrices for both
 * the activations and parameters.
 * For this use case, the "data" matrix represents the activations which are used
 * during the forward pass while the "grad" matrix represents the gradients
 * which are computed during the backward pass. Since these two matrices are typically
 * used together, it seems reasonable to create a single container class to manage them.
 * Some of the design of this class is based on the corresponding Variable class in Chainer:
 * http:\\chainer.org.
 *
 */
// Todo: The current implementation is not able to enforce that the 2 contained matrices always
// have the same extents. However, we can potentially enforce this if we do the following:
// Modify the Matrix class to share the extents of another Matrix instance B. Then when B's
// extents change, the shared matrix will also resize itself to have the same extents.
// The function to add would be share_extents(). We could then have grad.share_extents(data);
// Each matrix data and grad could share each other's extents, being careful not to introduce cycles.
template<typename T>
class Variable {

private:


public:

    // The data matrix
    Matrix<T> data;

    // The gradients matrix, which has the same size and extents as data.
    Matrix<T> grad;


    // We use the same constructors and resize() functions from Matrix, so please
    // refer to that class for documentation. These functions
    // perform the same operations on both the data and grad members.
    // Note that both data and grad are always initialized and/or resized to have the
    // same size.

    Variable()
    {}

    Variable(int e0):
        data{e0},
        grad{e0}
    {}

    Variable(int e0, int e1):
        data{e0, e1},
        grad{e0, e1}
    {}

    Variable(const std::vector<int> &extents):
        data{extents},
        grad{extents}
    {}

    Variable(int e0, int e1, int e2):
        data{e0, e1, e2},
        grad{e0, e1, e2}
    {}

    Variable(int e0, int e1, int e2, int e3):
        data{e0, e1, e2, e3},
        grad{e0, e1, e2, e3}
    {}

    Variable(int e0, int e1, int e2, int e3, int e4):
        data{e0, e1, e2, e3, e4},
        grad{e0, e1, e2, e3, e4}
    {}

    Variable(int e0, int e1, int e2, int e3, int e4, int e5):
        data{e0, e1, e2, e3, e4, e5},
        grad{e0, e1, e2, e3, e4, e5}
    {}

    void resize(const std::vector<int> &extents)
    {
        data.resize(extents);
        grad.resize(extents);
    }

    void resize(int e0) {
        data.resize(e0);
        grad.resize(e0);
    }

    void resize(int e0, int e1) {
        data.resize(e0, e1);
        grad.resize(e0, e1);
    }

    void resize(int e0, int e1, int e2) {
        data.resize(e0, e1, e2);
        grad.resize(e0, e1, e2);
    }

    void resize(int e0, int e1, int e2, int e3) {
        data.resize(e0, e1, e2, e3);
        grad.resize(e0, e1, e2, e3);
    }

    void resize(int e0, int e1, int e2, int e3, int e4) {
        data.resize(e0, e1, e2, e3, e4);
        grad.resize(e0, e1, e2, e3, e4);
    }

    void resize(int e0, int e1, int e2, int e3, int e4, int e5) {
        data.resize(e0, e1, e2, e3, e4, e5);
        grad.resize(e0, e1, e2, e3, e4, e5);
    }

    /**
     * Return the size of data or grad.
     */
    int size() const {
        return data.size();
    }

    /**
     * Return the size of the i'th extent (dimension) of data or grad..
     *
     * @param i If the i'th extent does not exist, return 0.
     */
    int extent(int i) const {
        return data.extent(i);
    }

    /**
     * Return a vector of extents for data or grad.
     *
     * The i'th component of the returned array contains the size
     * of the i'th dimension.
     *
     */
    const std::vector<int> &get_extents() const {
        return data.get_extents();
    }

    /**
     * Return the number of dimensions in data or grad.
     */
    int order() const {
        return data.order();
    }

};


using VariableF = Variable<float>;
using VariableD = Variable<double>;
using VariableI = Variable<int>;


}

#endif  /* _VARIABLE_H */
