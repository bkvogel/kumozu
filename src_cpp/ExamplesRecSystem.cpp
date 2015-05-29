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

#include "ExamplesRecSystem.h"
#include "NetflixRecSystem.h"
#include <cstdlib>
#include <string.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <ctime>
#include <omp.h>
#include <algorithm>

#include "MatrixT.h"
#include "MatrixIO.h"
#include "Matrix_list.h"
#include "Utilities.h"
#include "Constants.h"

// Uncomment following line to disable assertion checking.
//#define NDEBUG
// Uncomment to enable assertion checking.
#undef NDEBUG
#include <assert.h>

using namespace std;

namespace kumozu {

	void netflix_prize_example_1() {
		cout << "Netflix Prize Example 1" << endl << endl;
		// Path to training set
		const string training_set = "/home/brian/data/prototyping/netflix_data/download/training_set/";
		//const string training_set = "/Users/brian/data/prototyping/netflix_data/download/training_set/";
		// Path to probe file.
		const string probe = "/home/brian/data/prototyping/netflix_data/download/probe.txt";
		//const string probe = "/Users/brian/data/prototyping/netflix_data/download/probe.txt";

		NetflixRecSystem rec_system(training_set, probe);
		cout << "Learning parameters..." << endl;
		rec_system.learn_model();
		cout << "Done." << endl;
  }

  
}
