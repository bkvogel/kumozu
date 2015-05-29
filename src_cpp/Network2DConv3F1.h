#ifndef _NETWORK2DCONV3F1_H
#define _NETWORK2DCONV3F1_H
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


#include "MatrixT.h"
#include <string>
#include <iostream>
#include "Constants.h"
#include "Utilities.h"
#include "LinearLayer.h"
#include "ConvLayer2D.h"
#include "ConvLayer3D.h"
#include "ColumnActivationFunction.h"
#include "BoxActivationFunction.h"
#include "PoolingLayer.h"
#include "Dropout1D.h"
#include "Dropout3D.h"
#include "Network.h"

namespace kumozu {

	/*
	 *
	 * A neural network with 3 convolutional layers and 1 hidden fully-connected layer:
	 * 2D -> 3D convolutional layer -> pooling layer ->
	 * 3D -> 3D convolutional layer -> pooling layer ->
	 * 3D -> 3D convolutional layer -> pooling layer ->
	 * fully-connected hidden layer ->
	 * -> fully-connected output layer.
	 *
	 * The structure of the network is:
	 *
	 * input activations (a mini-batch of 2D greyscale images) ->  convolutional layer -> dropout -> max pooling layer 
	 *    -> convolutional layer -> dropout -> max pooling layer 
	 *    -> fully-connected ReLU layer -> dropout -> fully-connected ReLU layer -> output activations.
	 *
	 *
	 */
	class Network2DConv3F1 : public Network {

	public:

		/*
		* Create a new instance of this class.
		*
		* Parameters:
		*
		*
		*
		* dim_fully_connected_hidden: Number of hidden units between the two fully-connected layers.
		*/
	Network2DConv3F1(const std::vector<int>& input_extents, 
						  int filter_count1, int conv_filter_height1, int conv_filter_width1, 
							const std::vector<int>& pooling_region_extents1, const std::vector<int>& pooling_output_extents1,
						int filter_count2, int conv_filter_height2, int conv_filter_width2, 
							const std::vector<int>& pooling_region_extents2, const std::vector<int>& pooling_output_extents2,
						int filter_count3, int conv_filter_height3, int conv_filter_width3, 
							const std::vector<int>& pooling_region_extents3, const std::vector<int>& pooling_output_extents3,
							int dim_output,
							int dim_fully_connected_hidden, int maxout_factor,
							BoxActivationFunction::ACTIVATION_TYPE box_activation_type,
							ColumnActivationFunction::ACTIVATION_TYPE col_activation_type,
							const std::vector<float>& dropout_keep_probabilities) :
		Network(input_extents),
			m_conv_layer1(input_extents, filter_count1, conv_filter_height1, conv_filter_width1, "Conv_layer1"),
			m_dropout_conv_layer1 {m_conv_layer1.get_output_extents(), dropout_keep_probabilities.at(0)},
			m_box_activation1(m_dropout_conv_layer1.get_output_extents(), box_activation_type),
			m_pooling_layer1(m_box_activation1.get_output_extents(), pooling_region_extents1, pooling_output_extents1),
				m_conv_layer2(m_pooling_layer1.get_output_extents(), filter_count2, conv_filter_height2, conv_filter_width2, "Conv_layer2"),
				m_dropout_conv_layer2(m_conv_layer2.get_output_extents(), dropout_keep_probabilities.at(1)),
			m_box_activation2(m_dropout_conv_layer2.get_output_extents(), box_activation_type),
			m_pooling_layer2(m_box_activation2.get_output_extents(), pooling_region_extents2, pooling_output_extents2),
				m_conv_layer3(m_pooling_layer2.get_output_extents(), filter_count3, conv_filter_height3, conv_filter_width3 , "Conv_layer3"),
				m_dropout_conv_layer3(m_conv_layer3.get_output_extents(), dropout_keep_probabilities.at(2)),
			m_box_activation3(m_dropout_conv_layer3.get_output_extents(), box_activation_type),
			m_pooling_layer3(m_box_activation3.get_output_extents(), pooling_region_extents3, pooling_output_extents3),
			// Note: input_extents.at(0) is mini-batch size.
			m_full_layer1_input {  m_pooling_layer3.get_output_height()*m_pooling_layer3.get_output_width()*m_pooling_layer3.get_output_depth(), 
					input_extents.at(0)},
			m_full_layer1_input_error { m_full_layer1_input.extent(0) , m_full_layer1_input.extent(1)},
				m_full_layer1(m_full_layer1_input.get_extents(), dim_fully_connected_hidden*maxout_factor, "FC_layer1"),
					m_dropout_full_layer1(m_full_layer1.get_output_extents(), dropout_keep_probabilities.at(3)),
					m_full_activation_func1(m_dropout_full_layer1.get_output_extents(), dim_fully_connected_hidden,
											col_activation_type),
					m_full_layer2(m_full_activation_func1.get_output_extents(), dim_output*maxout_factor, "FC_layer2"),
					m_full_activation_func2(m_full_layer2.get_output_extents(), dim_output,
											col_activation_type)
			{
				std::cout << "Network2DConv3F1 info:" << std::endl;
				std::cout << "--------------------------------------" << std::endl;
				std::cout << "Mini-batch size = " << input_extents.at(0) << std::endl;
				std::cout << "Input activations image height = " << m_conv_layer1.get_height() << std::endl;
				std::cout << "Input activations image width = " << m_conv_layer1.get_width() << std::endl;
				std::cout << "--------------------------------------" << std::endl;
				std::cout << "Conv layer 1 input/output height = " << m_conv_layer1.get_height() << std::endl;
				std::cout << "Conv layer 1 input/output width = " << m_conv_layer1.get_width() << std::endl;
				std::cout << "Conv layer 1 output depth = " << m_conv_layer1.get_output_depth() << std::endl;
				std::cout << "--------------------------------------" << std::endl;
				std::cout << "Pooling layer 1 output height = " << m_pooling_layer1.get_output_height() << std::endl;
				std::cout << "Pooling layer 1 output width = " << m_pooling_layer1.get_output_width() << std::endl;
				std::cout << "Pooling layer 1 output depth = " << m_pooling_layer1.get_output_depth() << std::endl;
				std::cout << "--------------------------------------" << std::endl;
				std::cout << "Conv layer 2 input/output height = " << m_conv_layer2.get_height() << std::endl;
				std::cout << "Conv layer 2 input/output width = " << m_conv_layer2.get_width() << std::endl;
				std::cout << "Conv layer 2 input depth = " << m_conv_layer2.get_input_depth() << std::endl;
				std::cout << "Conv layer 2 output depth = " << m_conv_layer2.get_output_depth() << std::endl;
				std::cout << "--------------------------------------" << std::endl;
				std::cout << "Pooling layer 2 output height = " << m_pooling_layer2.get_output_height() << std::endl;
				std::cout << "Pooling layer 2 output width = " << m_pooling_layer2.get_output_width() << std::endl;
				std::cout << "Pooling layer 2 output depth = " << m_pooling_layer2.get_output_depth() << std::endl;
				std::cout << "--------------------------------------" << std::endl;
				std::cout << "Conv layer 3 input/output height = " << m_conv_layer3.get_height() << std::endl;
				std::cout << "Conv layer 3 input/output width = " << m_conv_layer3.get_width() << std::endl;
				std::cout << "Conv layer 3 input depth = " << m_conv_layer3.get_input_depth() << std::endl;
				std::cout << "Conv layer 3 output depth = " << m_conv_layer3.get_output_depth() << std::endl;
				std::cout << "--------------------------------------" << std::endl;
				std::cout << "Pooling layer 3 output height = " << m_pooling_layer3.get_output_height() << std::endl;
				std::cout << "Pooling layer 3 output width = " << m_pooling_layer3.get_output_width() << std::endl;
				std::cout << "Pooling layer 3 output depth = " << m_pooling_layer3.get_output_depth() << std::endl;
				std::cout << "--------------------------------------" << std::endl;
				

			}

			


		/*
		* Compute the output activations as a function of input activations.
		*/
		virtual void forward_propagate(const Matrix& input_activations);

		/*
		 * Perform back-propagation to compute all gradients: weights, biases, and activation deltas.
		 *
		 * Call forward_propagate() before this method to compute output activations.
		 *
		 * This updates m_W_grad and m_bias_grad.
		 */
		virtual void back_propagate_gradients(const Matrix& input_activations, 
													 const Matrix& true_output_activations);

		/*
		 * Perform back-propagation to compute all activation deltas including the input layer deltas.
		 *
		 * Call forward_propagate() before this method to compute output activations.
		 *
		 * This updates the "delta activations" for all layers, including the supplied input_activations_deltas
		 * for the input layer.
		 * This method is intended to be used to infer input activations when the input is partially or
		 * completely hidden.
		 *
		 */
		void back_propagate_deltas(Matrix& input_activations_deltas, 
													 const Matrix& true_output_activations);

		/*
		 * Compute the cost function J(W, bias, true_output_activations, current input activations)
		 *
		 * where J = 0.5*(||actual_output_activations - true_output_activations||^2)
		 *
		 * true_output_activations: The desired output activations, such as the training activations values.
		 * 
		 */ 
		virtual float compute_cost_function(const Matrix& input_activations, const Matrix& true_output_activations);

		/*
		 * Update weights and bias parameters.
		 *
		 * This does not compute the gradients. This method should therefore be called after calling
		 * compute_gradients().
		 */
		virtual void update_weights();



		/*
		 * Perform numerical gradient checks using finite differences method and compare to
		 * gradients computed by back-propagation. Return true if the checks pass. Otherwise,
		 * return false.
		 */
		virtual bool check_gradients(const Matrix& input_activations, const Matrix& true_output_activations);

		/*
		 * Return a reference to this layer's output activations.
		 */
		virtual const Matrix& get_output() const {
			return m_full_activation_func2.get_output();
		}

		/*
		 * Return a reference to this layer's output activations.
		 */
		virtual Matrix& get_output() {
			return m_full_activation_func2.get_output();
		}

		/*
		 * Enable dropout.
		 *
		 * Dropout should only be enabled in "train mode."
		 * Dropout should be disabled before running in "test mode."
		 */
		virtual void enable_dropout() {
			m_dropout_conv_layer1.enable_dropout();
			m_dropout_conv_layer2.enable_dropout();
			m_dropout_conv_layer3.enable_dropout();
			m_dropout_full_layer1.enable_dropout();
		}

		/*
		 * Disable dropout. 
		 *
		 * Dropout should only be enabled in "train mode."
		 * Dropout should be disabled before running in "test mode."
		 *
		 * To enable again, call enable_dropout().
		 */
		virtual void disable_dropout() {
			m_dropout_conv_layer1.disable_dropout();
			m_dropout_conv_layer2.disable_dropout();
			m_dropout_conv_layer3.disable_dropout();
			m_dropout_full_layer1.disable_dropout();
		}

		/*
		 * Save training and test error to a file withe the prefix given
		 * by the supplied name.
		 */
		virtual void save_learning_info(std::string name) const;
		
		/*
		 * Load learned parameters from a file. The string name should be
		 * the same that was used to save the parameters when
		 * save_learning_info() was called.
		 */
		virtual void load_parameters(std::string name) const;

	// Make private again after debug.
	//private:

		ConvLayer2D m_conv_layer1;
		Dropout3D m_dropout_conv_layer1;
		BoxActivationFunction m_box_activation1;
		PoolingLayer m_pooling_layer1;
		ConvLayer3D m_conv_layer2;
		Dropout3D m_dropout_conv_layer2;
		BoxActivationFunction m_box_activation2;
		PoolingLayer m_pooling_layer2;
		ConvLayer3D m_conv_layer3;
		Dropout3D m_dropout_conv_layer3;
		BoxActivationFunction m_box_activation3;
		PoolingLayer m_pooling_layer3;
		Matrix m_full_layer1_input;
		Matrix m_full_layer1_input_error;
		LinearLayer m_full_layer1;
		Dropout1D m_dropout_full_layer1;
		ColumnActivationFunction m_full_activation_func1;
		LinearLayer m_full_layer2;
		ColumnActivationFunction m_full_activation_func2;

	};

}

#endif /* _NETWORK2DCONV3F1_H */
