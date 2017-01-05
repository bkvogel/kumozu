#ifndef _NMF_UPDATER_LEE_SEUNG_H
#define _NMF_UPDATER_LEE_SEUNG_H
/*
 * Copyright (c) 2005-2016, Brian K. Vogel
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

#include "Matrix.h"
#include <string>
#include <iostream>
#include <functional>
#include <map>
#include "Constants.h"
#include "Utilities.h"

namespace kumozu {

  class NmfUpdaterLeeSeung {

  public:

    /**
     * Create an instance of an NMF updater that performs the factorization:
     *
     * X approx= W * H
     *
     * This class implements slightly modified versions of the multiplicative update rules from 
     * "Algorithms for non-negative matrix factorization" by Lee and Seung, referred to as [1] below.
     *
     * The modified version uses my method of controlling the strength of an update so 
     * that the Lee-Seung update rules can
     * be used in an on-line setting, such as mini-batch training.
     *
     */
  NmfUpdaterLeeSeung(std::string name) :
    m_name {name}
    {
    }



    /**
     * Update H in the factorization:
     *
     * X approx= W * H
     *
     *
     * Perform the Lee and Seung update for H, weighted by alpha_H:
     *
     * H[i,k] <- H[i,k] *(alpha_H*gamma_H + (1-alpha_H))
     *
     * where alpha_H is in the range [0, 1]
     *
     * and where gamma_H is defined as
     *
     *
     *              [W^T * X] [i,k] + epsilon
     * gamma_H =  --------------------------------
     *              [W^T * (W * H)][i,j] + epsilon
     *
     * and where epsilon is a small positive constant to prevent division by zero.
     *
     * Note that for alpha_H=1 and epsilon=0, the above rule reduces exactly to the update rule from [1].
     */
  void right_update(const MatrixF& X, const MatrixF& W, MatrixF& H, float alpha=1.0f);

    /**
     * Update W in the factorization:
     *
     * X approx= W * H
     *
     * Perform the Lee and Seung update for W, weighted by alpha_W:
     *
     * W[i,k] <- W[i,k] *(alpha_W*gamma_W + (1-alpha_W))
     *
     * where alpha_W is in the range [0, 1]
     *
     * and where gamma_W is defined as
     *
     *
     *              [X * H^T] [i,k] + epsilon
     * gamma_W =  --------------------------------
     *              [(W * H) * H^T][i,j] + epsilon
     *
     * and where epsilon is a small positive constant to prevent division by zero.
     *
     * Note that for alpha_W=1 and epsilon=0, the above rule reduces exactly to the update rule from [1].
     */
    void left_update(const MatrixF& X, MatrixF& W, const MatrixF& H, float alpha=1.0f);

    /**
     * Update X as the weighted product:
     *
     * X <- alpha_X*(W*H) + (1-alpha_X)*X
     *
     * Note that for alpha_X = 1, this reduces to:
     *
     * X <- W * H
     *
     * and that for alpha_X = 0, this reduces to:
     *
     * X <- X (no change).
     *
     */
    void product_update(MatrixF& X, const MatrixF& W, const MatrixF& H, float alpha=1.0f);


    /**
     * Compute the root mean squared error (RMSE) for the factorization approximation and return
     * it.
     *
     * @param skip_product If true, skip the computation of X_approx = W*H for faster performance. The
     * most recently computed X_approx values will then be used.
     * This is only safe if neither
     * W nor H has been modified since the last time another member function has been called with
     * skip_product=false.
     * @return The current RMSE per element of X.
     */
    float get_rmse(MatrixF& X, const MatrixF& W, const MatrixF& H);

    
    /**
     * Print the name of this updater along with the current RMSE.
     */
    void print_rmse(MatrixF& X, const MatrixF& W, const MatrixF& H);

    
    /**
     * Prevent the values in the parameter matrices W and H from getting too close to zero.
     *
     * If a value less than \p epsilon is found, it will be replaced by a
     * value in [epsilon/2, epsilon]. This limits the smaller possible value to epsilon/2.
     * This is performed using a simple linear mapping from [0, epsilon] to [epsilon/2, epsilon].
     *
     * This serves two purposes:
     * 1. Extremely small values (denornamlized values) are typically much slower to compute
     * by hardware floating point units. Preventing denormals can speed up computation.
     * 2. Once a value becomes 0, it is impossible for it to ever become non-zero again, since
     * the Lee-Seung algorithm uses multiplicative updates.
     *
     * @param epsilon Any values in W or H that are less than this value will be replaced
     * by a value of at least epsilon/2 using the described mapping.
     *
     */
    void set_flag_prevent_zero(float epsilon);


  private:

    std::string m_name;
    float m_epsilon {1e-7f};
    MatrixF m_temp_size_W_tran;
    MatrixF m_temp_size_H;
    MatrixF m_temp_size_X;
    MatrixF m_temp_size_H2;
    MatrixF m_temp_size_H3;
    MatrixF m_temp_size_H_tran;
    MatrixF m_temp_size_W;
    MatrixF m_temp_size_W2;
    MatrixF m_temp_size_W3;
    MatrixF m_temp_size_X2;
    MatrixF m_temp_size_H4;
    MatrixF m_temp_size_W4;
    float m_prevent_zero_epsilon {1e-7f};
  };

}

#endif /* _NMF_UPDATER_LEE_SEUNG_H */
