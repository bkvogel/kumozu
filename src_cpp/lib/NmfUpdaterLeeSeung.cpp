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
#include "NmfUpdaterLeeSeung.h"
#include "Utilities.h"

using namespace std;

namespace kumozu {

  
  void NmfUpdaterLeeSeung::right_update(const MatrixF& X, const MatrixF& W, MatrixF& H, float alpha) {
    transpose(m_temp_size_W_tran, W); // W^T
    mat_multiply(m_temp_size_H, m_temp_size_W_tran, X); // m_temp_size_H is numerator
    mat_multiply(m_temp_size_X, W, H); // m_temp_size_X is prediction W*H
    mat_multiply(m_temp_size_H2, m_temp_size_W_tran, m_temp_size_X); // m_temp_size_H2 is denominator
    element_wise_divide(m_temp_size_H3, m_temp_size_H, m_temp_size_H2, m_epsilon); // m_temp_size_H3 is gamma_H
    scale(m_temp_size_H3, alpha); // Scale gamm_H by alpha
    const float one_minus_alpha = 1.0f - alpha;
    //add_scalar(m_temp_size_H3, one_minus_alpha); // m_temp_size_H3 now contains: alpha*gamma_H + (1-alpha)
    //copy_matrix(m_temp_size_H4, H);
    m_temp_size_H4 = H;
    scale(m_temp_size_H4, one_minus_alpha);
    element_wise_multiply(H, H, m_temp_size_H3);  // H[i,k] <- H[i,k] *(alpha*gamma_H + (1-alpha))
    element_wise_sum(H, m_temp_size_H4, H);

    // Replace any extremely small values by values of at least m_prevent_zero_epsilon/2.
    apply(H, [=] (float a) {
        if (a < m_prevent_zero_epsilon) {
            return 0.5f*a + m_prevent_zero_epsilon/2.0f;
        } else {
            return a;
        }
    });
  }

  
  void NmfUpdaterLeeSeung::left_update(const MatrixF& X, MatrixF& W, const MatrixF& H, float alpha) {
    transpose(m_temp_size_H_tran, H); // H^T
    mat_multiply(m_temp_size_W, X, m_temp_size_H_tran); // m_temp_size_W is numerator
    mat_multiply(m_temp_size_X, W, H); // m_temp_size_X is prediction W*H
    mat_multiply(m_temp_size_W2, m_temp_size_X, m_temp_size_H_tran); // m_temp_size_W2 is denominator
    element_wise_divide(m_temp_size_W3, m_temp_size_W, m_temp_size_W2, m_epsilon); // m_temp_size_W3 is gamma_W
    scale(m_temp_size_W3, alpha); // Scale gamma_W by alpha
    const float one_minus_alpha = 1.0f - alpha;
    //add_scalar(m_temp_size_W3, one_minus_alpha); // m_temp_size_W3 now contains: alpha*gamma_W + (1-alpha)
    //scale(W, one_minus_alpha);
    //copy_matrix(m_temp_size_W4, W);
    m_temp_size_W4 = W;
    scale(m_temp_size_W4, one_minus_alpha);
    element_wise_multiply(W, W, m_temp_size_W3);  // W[i,k] <- W[i,k] *(alpha*gamma_W + (1-alpha))
    element_wise_sum(W, m_temp_size_W4, W);

    // Replace any extremely small values by values of at least m_prevent_zero_epsilon/2.
    apply(W, [=] (float a) {
        if (a < m_prevent_zero_epsilon) {
            return 0.5f*a + m_prevent_zero_epsilon/2.0f;
        } else {
            return a;
        }
    });
  }

  
  void NmfUpdaterLeeSeung::product_update(MatrixF& X, const MatrixF& W, const MatrixF& H, float alpha) {
    mat_multiply(m_temp_size_X, W, H);
    scale(m_temp_size_X2, m_temp_size_X, alpha);
    const float one_minus_alpha = 1 - alpha;
    scale(X, one_minus_alpha);
    element_wise_sum(X, m_temp_size_X2, X);
  }


  float NmfUpdaterLeeSeung::get_rmse(MatrixF& X, const MatrixF& W, const MatrixF& H) {
    mat_multiply(m_temp_size_X, W, H); // m_temp_size_X is prediction W*H
    map2(m_temp_size_X2, m_temp_size_X, X, [] (float a, float b) {
	return std::abs(a-b)*std::abs(a-b);
      });
    const float N = static_cast<float>(X.size());
    const float rmse = std::sqrt(sum(m_temp_size_X2) / N);
    return rmse;
  }

  
  void NmfUpdaterLeeSeung::print_rmse(MatrixF& X, const MatrixF& W, const MatrixF& H) {
    cout << m_name << ": RMSE: " << get_rmse(X, W, H) << endl;
  }

  
  void NmfUpdaterLeeSeung::set_flag_prevent_zero(float epsilon) {
      m_prevent_zero_epsilon = epsilon;
  }
  
}
