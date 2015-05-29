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

#include "Network.h"
#include "Utilities.h"

using namespace std;

namespace kumozu {

	void Network::numerical_gradients(const Matrix& input_activations, 
								 const Matrix& true_output_activations, const Matrix& gradients_back_prop, 
								 Matrix& parameters) {
			Matrix gradients_numerical = gradients_back_prop; // Temp matrix to hold the numerical gradients.
			set_value(gradients_numerical, 0.0f);
			for (int n = 0; n != parameters.size(); ++n) {
				float orig = parameters[n]; // backup
				parameters[n] += m_epsilon;
				// Now compute J(theta_plus)
				float J_plus = compute_cost_function(input_activations, true_output_activations);
				// Now compute J(theta_minus)
				parameters[n] = orig - m_epsilon;
				float J_minus = compute_cost_function(input_activations, true_output_activations);
				// Put back original value.
				parameters[n] = orig;
				gradients_numerical[n] = (J_plus - J_minus)/(2*m_epsilon);
			}
			const float relative_error_score = relative_error(gradients_back_prop, gradients_numerical);
			std::cout << "numerical-back-prop gradients relative error = " << relative_error_score << std::endl;
			std::cout << "gradients_back_prop = " << std::endl << gradients_back_prop << std::endl;
			std::cout << "-----------------------------" << std::endl;
			std::cout << "gradients_numerical = " << std::endl << gradients_numerical << std::endl;
			assert_almost_equal(relative_error_score, 0.0f, m_pass_relative_error);
		}



}
