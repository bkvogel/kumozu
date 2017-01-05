#ifndef _ACCUMULATOR_H
#define _ACCUMULATOR_H
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

#include "Matrix.h"
#include <string>
#include <iostream>

namespace kumozu {

  /*
   * A utility class to keep track of a running sum or mean value.
   *
   * Usage:
   *
   * An example of
   */
  class Accumulator {
  public:

  Accumulator():
    m_sum {0},
      m_batch_size {1},
        m_counter {0}
        {

        }

  Accumulator(int batch_size):
        m_sum {0},
          m_batch_size {batch_size},
            m_counter {0}
            {

            }

            /*
             * Accumulate the supplied value and increment the internal counter by the "batch_size"
             * that was supplied to the constructor, or by 1 if the default constructor was used.
             */
            void accumulate(float x);

            /*
             * Reset the accumulator.
             *
             * This sets the internal sum and counter to 0.
             */
            void reset();

            /*
             * Return the sum of all values that have been passed to accumulate() sence the last time
             * reset() was called.
             */
            float get_sum() const;

            /*
             * Return the sum of all values that have been passed to accumulate() divided by
             * the counter value sence the last time
             * reset() was called.
             */
            float get_mean() const;

            /*
             * Return the value of the internal counter.
             */
            int get_counter() const;

  private:

            float m_sum;
            int m_batch_size;
            int m_counter;

  };


}

#endif /* _ACCUMULATOR_H */
