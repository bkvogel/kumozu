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
#include "Dropout1D.h"

#include "Utilities.h"
using namespace std;

namespace kumozu {

	void Dropout1D::forward_dropout(const Matrix& input) {
		check_dimensions(input, m_output);
		// Compute a new random dropout mask to use on each column in the mini-batch.
		std::uniform_real_distribution<float> uni(0.0f, 1.0f);
		for (size_t i = 0; i < m_dropout_mask.size(); i++) {
			float x = uni(m_mersenne_twister_engine);
			//m_temp_rand[i] = x; // fixme: experiemental
			m_temp_rand[i] = 1-x; // fixme: experiemental
			if (x < m_prob_keep_current) {
				// Keep the element
				m_dropout_mask[i] = 1;
			} else {
				// Drop the element
				m_dropout_mask[i] = 0;
			}
		}
		// Apply the dropout mask.
		int minibatch_size = m_output.extent(1);
		#pragma omp parallel for collapse(2)
		for (int n = 0; n < m_output.extent(0); ++n) {
			/*
			for (int m = 0; m < minibatch_size; ++m) {
				if (m_dropout_mask[n] == 1) {
					m_output(n, m) = input(n, m)/m_prob_keep_current; // inverted dropout
					//m_output(n, m) = input(n, m);
				} else {
					m_output(n, m) = 0.0f;
				}
			}
			*/
			
			for (int m = 0; m < minibatch_size; ++m) {
				if (m_dropout_mask[n] == 1) {
					m_output(n, m) = input(n, m)/m_prob_keep_current; // inverted dropout
					//m_output(n, m) = input(n, m);
				} else {
					m_output(n, m) = 0.0f; // works well
					// leaky dropout
					//m_output(n, m) = 0.1f*input(n, m);
					// random leaky dropout
					//m_output(n, m) = m_temp_rand[n]*input(n, m);
				}
			}
			
		}
		
	}

	void Dropout1D::reverse_dropout(Matrix& input) {
		check_dimensions(input, m_output);
		// Apply the dropout mask.
		int minibatch_size = m_output.extent(1);
		#pragma omp parallel for collapse(2)
		for (int n = 0; n < m_output.extent(0); ++n) {
			for (int m = 0; m < minibatch_size; ++m) {
				/*
				if (m_dropout_mask[n] == 1) {
					input(n, m) = m_output_deltas(n, m)/m_prob_keep_current; // for inverted dropout
					//input(n, m) = m_output_deltas(n, m);
				} else {
					input(n, m) = 0.0f;
				}
				*/
				
				if (m_dropout_mask[n] == 1) {
					//input(n, m) = m_output_deltas(n, m)*m_prob_keep_current; // for inverted dropout
					input(n, m) = m_output_deltas(n, m)/m_prob_keep_current;
				} else {
					input(n, m) = 0.0f; // works well
					// Leaky dropout
					//input(n, m) = 0.1f*m_output_deltas(n, m);
					// Random leaky dropout
					//input(n, m) = m_temp_rand[n]*m_output_deltas(n, m);
				}
				
			}
		}
		
	}



}
