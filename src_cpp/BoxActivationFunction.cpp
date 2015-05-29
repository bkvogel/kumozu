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
#include "BoxActivationFunction.h"

#include "Utilities.h"
using namespace std;

namespace kumozu {

	void BoxActivationFunction::forward_activation(const Matrix& input) {
		if (m_activation_type == ACTIVATION_TYPE::ReLU) {
			compute_forward_relu(input, m_output, m_state); 
		} else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
			compute_forward_leaky_relu(input, m_output, m_state); 
		} else if (m_activation_type == ACTIVATION_TYPE::linear) {
			compute_forward_identity_activation(input, m_output, m_state); 
		} else if (m_activation_type == ACTIVATION_TYPE::kmax) {
			compute_forward_3d_kmax(input, m_output, m_state, m_box_depth, m_box_height, m_box_width, m_k);
		}
	}

	void BoxActivationFunction::reverse_activation(Matrix& input) {
		if (m_activation_type == ACTIVATION_TYPE::ReLU) {
			compute_reverse_relu(input, m_output_deltas, m_state); 
		} else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
			compute_reverse_leaky_relu(input, m_output_deltas, m_state); 
		} else if (m_activation_type == ACTIVATION_TYPE::linear) {
			compute_reverse_identity_activation(input, m_output_deltas, m_state); 
		} else if (m_activation_type == ACTIVATION_TYPE::kmax) {
			compute_reverse_3d_kmax(input, m_output_deltas, m_state);
		}
	}

}
