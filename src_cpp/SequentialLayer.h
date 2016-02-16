#ifndef _SEQUENTIAL_LAYER_H
#define _SEQUENTIAL_LAYER_H
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
#include "PlotUtilities.h"
#include "Node.h"
#include <functional>

namespace kumozu {

  /*
   * This is a Node that is a container for a sequence of Layer nodes.
   *
   * This class corresponds to a composite node with 1 input port and 1 output port in which one or more 
   * contained layers are connected serially to form a feed-forward graph.
   * 
   * It is possible to nest instances of this class, such that one or more of the contained nodes may also
   * be an instance of SequentialLayer.
   *
   * Usage:
   *
   * Create an instance of this class and add layers using the add_layer() function.
   *
   *
   */
  class SequentialLayer : public Node {

  public:

  SequentialLayer(std::string name) :
    Node(name)
    {

    }

    virtual ~SequentialLayer(){}

    /*
     * Add a node to the sequence of contained nodes.
     *
     * Nodes are connected in series in the
     * order that they were added. 
     *
     * The supplied node must already have exactly 1 output port. This function
     * will give it 1 input port as it is connected inside this node.
     */
    void add_layer(Node& contained_node);


  protected:

    /*
     * This does nothing.
     */
    virtual void reinitialize() {}

  private:


  };

}


#endif /* _SEQUENTIAL_LAYER_H */
