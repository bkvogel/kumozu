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
#include "Matrix_list.h"
#include<sstream>
using namespace std;

namespace kumozu {
	Matrix_list::Matrix_list(const Matrix& A) {
		rows = A.extent(0);
		columns = A.extent(1);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				Matrix_element element;
				element.row = r;
				element.col = c;
				element.val = A.get(r, c);
				observed_list.push_back(element);
			}
		}
	}

	Matrix_list::Matrix_list(int row_count, int col_count) {
		rows = row_count;
		columns = col_count;
	}

	//Matrix_list::~Matrix_list() {
	//}


	std::ostream& operator<<(std::ostream& os, const Matrix_list& m) {
		os << "Row dimension = " << m.rows << endl;
		os << "Column dimension = " << m.columns << endl;
		os << "Size = " << m.observed_list.size() << endl;
		os << "Elements:" << endl;
		for (vector<Matrix_element>::const_iterator iter = m.observed_list.begin(); iter != m.observed_list.end(); ++iter) {
			os << "-----------------------------------" << endl;
			os << "row = " << iter->row << endl;
			os << "col = " << iter->col << endl;
			os << "val = " << iter->val << endl;
			os << "-----------------------------------" << endl;
		}
		return os;
	}

}
