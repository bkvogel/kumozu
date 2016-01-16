#ifndef _PLOTUTILITIES_H
#define _PLOTUTILITIES_H
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
#include <iostream>
#include <vector>
#include "gnuplot-iostream.h"

namespace kumozu {

  // This file contains a collection of functions that partially wrap the "gnuplot-iostream" interface to Gnuplot, as well
  // as some convinience utilities for converting a matrix containing multiple images into a suitable plotable
  // image.
  //
  // There are two versions of most of the function in this file:
  //
  // One version requires that the user get a Gnuplot instance first before calling the
  // function. This version should be used if you would like to be able to keep redrawing
  // the plot in the same figure window as new or different data becomes available, or if you would like like to use
  // the "multiplot" feature to put multiple plots into the same window.
  //
  // The other version is slightly simpler to use since the user does not need to pass in a Gnuplot reference.
  // A new window will be created each time the function is called.
  // This version should be used in the typical case in which the data is only plotted once. That is, once the
  // plot is drawn, it can no longer be updated with new data.


  /*
   * Plot the supplied vector and use the supplied title.
   *
   * This version requires that the user get a Gnuplot instance first before calling this
   * function. This version should be used if you would like to be able to keep redrawing
   * the plot in the same figure window as new or different data becomes available, or if you would like like to use
   * the "multiplot" feature to put multiple plots into the same window.
   *
   * The title is required, but any other Gnuplot parameters are optional and can
   * be passed in the "options" parameter. The "options" parameter is a vector of strings
   * where each string in the vector represents one option. For example, to supply
   * x and y labels for the plot, we can create the options as follows:
   *
   * vector<string> options = {"set xlabel 'elapsed time, minutes'", "set ylabel 'hotdogs per minute'"};
   *
   */
  template <typename T>
    void plot(Gnuplot& gp, const std::vector<T>& vec, std::string title, std::vector<std::string> options = {}) {
    gp << "set title '" + title + "'" << std::endl;
    for (auto one_option : options) {
      gp << one_option << std::endl;
    }
    //gp << "plot '-' with lines notitle\n";
    //gp.send1d(vec);
    gp << "plot '-' binary" << gp.binFmt1d(vec, "array") << "with lines notitle" << std::endl;
    //gp.send1d(vec);
    gp.sendBinary1d(vec);
  }

  /*
   * Plot the supplied vector and use the supplied title.
   *
   * Note: A new window will be created every time this function is called. If you do not want
   * this behavior, call the version of this function that takes a Gnuplot reference instead.
   *
   * The title is required, but any other Gnuplot parameters are optional and can
   * be passed in the "options" parameter. The "options" parameter is a vector of strings
   * where each string in the vector represents one option. For example, to supply
   * x and y labels for the plot, we can create the options as follows:
   *
   * vector<string> options = {"set xlabel 'elapsed time, minutes'", "set ylabel 'hotdogs per minute'"};
   *
   */
  template <typename T>
    void plot(const std::vector<T>& vec, std::string title, std::vector<std::string> options = {}) {
    Gnuplot gp;
    plot(gp, vec, title, options);
  }


  /*
   * Plot a 2D matrix as a greyscale image. The supplied
   * "image" must have size (height, width).
   * The assumed range of pixel values is [0,255].
   *
   * This version requires that the user get a Gnuplot instance first before calling this
   * function. This version should be used if you would like to be able to keep redrawing
   * the plot in the same figure window as new or different data becomes available, or if you would like like to use
   * the "multiplot" feature to put multiple plots into the same window.
   */
  template <typename T>
    void plot_image_greyscale(Gnuplot& gp, const Matrix<T>& image, std::string title, std::vector<std::string> options = {}) {
    if (image.order() == 1) {
      // Hack to get it to plot a 1-d matrix as an image, which requires at least 2x2 image in Gnuplot.
      // Convert to a image.order() x 2 matrix with duplicated columns.
      Matrix<T> temp(image.size(), 2);
      for (int n = 0; n < image.size(); ++n) {
        for (int c = 0; c < 2; ++c) {
          temp(n, c) = image(n);
        }
      }
      // Call again with proper sized iamge.
      return plot_image_greyscale(gp, temp, title, options);
    }

    // Error checking:
    if (image.order() != 2) {
      std::cerr << "plot_image_greyscale(): Image has wrong order. Should be 2." << std::endl;
      exit(1);
    }

    const int height = image.extent(0);
    const int width = image.extent(1);
    std::vector<std::vector<T> > im;
    for(int i=0; i<height; i++) {
      std::vector<T> row;
      for(int j=0; j<width; j++) {
        T val = image(i,j);
        row.push_back(val);
      }
      im.push_back(row);
    }

    gp << "set title '" + title + "'\n";
    gp << "set palette grey\n"; // colomap
    //gp << "set palette color\n"; // colomap
    //gp << "set palette model XYZ rgbformulae 7,5,15\n"; // hot colormap
    gp << "set border linewidth 0\n";
    gp << "unset colorbox\n";
    gp << "unset tics\n";
    gp << "unset key\n";

    for (auto one_option : options) {
      //std::cout << one_option << std::endl;
      gp << one_option << "\n";
    }

    //gp << "plot '-' binary" << gp.binFmt2d(im, "array") << " with image\n";
    gp << "plot '-' binary" << gp.binFmt2d(im, "array") << " flipy with image\n";
    gp.sendBinary2d(im);

  }


  /*
   * Plot a 2D matrix as a greyscale image. The supplied
   * "image" must have size (height, width).
   * The assumed range of pixel values is [0,255].
   *
   * Note: A new window will be created every time this function is called. If you do not want
   * this behavior, call the version of this function that takes a Gnuplot reference instead.
   */
  template <typename T>
    void plot_image_greyscale(const Matrix<T>& image, std::string title, std::vector<std::string> options = {}) {
    Gnuplot gp;
    plot_image_greyscale(gp, image, title, options);
  }

  /*
   * Plot an RGB image. The supplied "image" should have dimensions
   * (depth, height, width).
   * where depth = 3 so that (0,i,j) represents red value of pixel (i,j)
   * (1,i,j) represents green value, and (2,i,j) represents blue value.
   *
   * The values of the elements of "image" should be in the range [0, 255].
   * Any numeric type that can be cast to an unsigned char is allowed.
   *
   * This version requires that the user get a Gnuplot instance first before calling this
   * function. This version should be used if you would like to be able to keep redrawing
   * the plot in the same figure window as new or different data becomes available, or if you would like like to use
   * the "multiplot" feature to put multiple plots into the same window.
   */
  template <typename T>
    void plot_image_rgb(Gnuplot& gp, const Matrix<T>& image, std::string title) {
    // Error checking:
    if (image.order() != 3) {
      std::cerr << "plot_image_rgb(): Image has wrong order. Should be 3." << std::endl;
      exit(1);
    }

    const int depth = image.extent(0);
    const int height = image.extent(1);
    const int width = image.extent(2);
    if (depth != 3) {
      std::cerr << "plot_image_rgb(): Image has wrong number of color channels. Should be 3." << std::endl;
      exit(1);
    }

    //std::cout << "depth = " << depth << std::endl;
    //std::cout << "height = " << height << std::endl;
    //std::cout << "width = " << width << std::endl;

    std::vector<std::vector<std::vector<unsigned char> > > im;
    for(int i=0; i<height; i++) {
      std::vector<std::vector<unsigned char> > row;
      for(int j=0; j<width; j++) {
        std::vector<unsigned char> pixels;
        for (int d=0; d < depth; ++d) {
          unsigned char val = static_cast<unsigned char>(image(d,i,j));
          pixels.push_back(val);
        }
        row.push_back(pixels);
      }
      im.push_back(row);
    }

    gp << "set title '" + title + "'\n";
    gp << "set border linewidth 0\n";
    gp << "unset colorbox\n";
    gp << "unset tics\n";
    gp << "unset key\n";
    // Note: "flipy" flips the plot so that the (0,0) corner is at the upper left of the plot
    // instead of the lower left.
    gp << "plot '-' binary" << gp.binFmt2d(im, "array") << " flipy with rgbimage\n";
    gp.sendBinary2d(im);
  }


  /*
   * Plot an RGB image. The supplied "image" should have dimensions
   * (depth, height, width).
   * where depth = 3 so that (0,i,j) represents red value of pixel (i,j)
   * (1,i,j) represents green value, and (2,i,j) represents blue value.
   *
   * The values of the elements of "image" should be in the range [0, 255].
   * Any numeric type that can be cast to an unsigned char is allowed.
   *
   * Note: A new window will be created every time this function is called. If you do not want
   * this behavior, call the version of this function that takes a Gnuplot reference instead.
   */
  template <typename T>
    void plot_image_rgb(const Matrix<T>& image, std::string title) {
    Gnuplot gp;
    plot_image_rgb(gp, image, title);
  }


  /*
   * Plot the images in the supplied "images" matrix.
   *
   * The supplied "images" matrix must have order 4 where the last 2 dimensions are image height and
   * image width. That is, the allowable size for "images" is:
   *
   * (dim0, dim1, height, width)
   * in which case there are dim0*dim1 images in total.
   *
   * An example usage of this function is when "images" corresponds to the filter weights in a convolutional neural
   * network. In this case, dim0 is the "filter count" and dim1 is "image depth".
   *
   * Since there is no RGB channel dimension, the images are assumed to be greyscale of size (height, width).
   *
   * The images are arranged into a rectangular grid of size dim0 x dim1 images where each image has size (height, width).
   *
   * This version requires that the user get a Gnuplot instance first before calling this
   * function. This version should be used if you would like to be able to keep redrawing
   * the plot in the same figure window as new or different data becomes available, or if you would like like to use
   * the "multiplot" feature to put multiple plots into the same window.
   */
  template <typename T>
    void plot_images_greyscale_4dim(Gnuplot& gp, const Matrix<T>& images, std::string title, std::vector<std::string> options = {}) {
    if (images.order() != 4) {
      std::cerr << "plot_images_greyscale_4dim(): supplied images matrix is wrong order." << std::endl;
      exit(1);
    }
    const int plot_rows = images.extent(0); // dim0
    const int plot_cols = images.extent(1); // dim1
    const int height = images.extent(2);
    const int width = images.extent(3);
    //std::cout << "plot_rows = " << plot_rows << std::endl;
    //std::cout << "plot_cols = " << plot_cols << std::endl;
    //std::cout << "height = " << height << std::endl;
    //std::cout << "width = " << width << std::endl;
    const int border_thickness = 1; // Thickness of border between images.
    const int out_image_rows = plot_rows*height + plot_rows*border_thickness;
    const int out_image_cols = plot_cols*width + plot_cols*border_thickness;
    //std::cout << "out_image_rows = " << out_image_rows << std::endl;
    //std::cout << "out_image_cols = " << out_image_cols << std::endl;
    Matrix<T> out_image(out_image_rows, out_image_cols);

    // We want the border to appear black (or same as the minimum value).
    //float minval = min_value(images);
    //std::cout << "minval = " << minval << std::endl;
    //set_value(out_image, minval);
    // Or we can set the border to appear white (or same as the max value)
    float maxval = max_value(images);
    set_value(out_image, maxval);

    for (int p_row = 0; p_row != plot_rows; ++p_row) {
      for (int p_col = 0; p_col != plot_cols; ++p_col) {
        // Calculate current offset from the (0,0) corner of the output plot image.
        int row_offset = p_row*(height + border_thickness);
        int col_offset = p_col*(width + border_thickness);
        for (int i = 0; i != height; ++i) {
          for (int j = 0; j != width; ++j) {
            out_image(row_offset + i, col_offset + j) = images(p_row, p_col, i,j);
            //out_image(row_offset + i, col_offset + j) = std::abs(images(p_row, p_col, i,j));
          }
        }
      }
    }
    // Want to keep aspect ratio so that pixels remain square-shaped:
    float ratio = static_cast<float>(plot_rows)/static_cast<float>(plot_cols);
    std::ostringstream cmd_str;
    cmd_str << "set size ratio " << ratio;
    options.push_back(cmd_str.str());
    //std::cout << "ratio command: " << cmd_str.str() << std::endl;
    // Now plot it.
    plot_image_greyscale(gp, out_image, title, options);


  }

  /*
   * Plot the images in the supplied "images" matrix.
   *
   * The supplied "images" matrix must have order 4 where the last 2 dimensions are image height and
   * image width. That is, the allowable size for "images" is:
   *
   * (dim0, dim1, height, width)
   * in which case there are dim0*dim1 images in total.
   *
   * An example usage of this function is when "images" corresponds to the filter weights in a convolutional neural
   * network. In this case, dim0 is the "filter count" and dim1 is "image depth".
   *
   * Since there is no RGB channel dimension, the images are assumed to be greyscale of size (height, width).
   *
   * The images are arranged into a rectangular grid of size dim0 x dim1 images where each image has size (height, width).
   *
   * Note: A new window will be created every time this function is called. If you do not want
   * this behavior, call the version of this function that takes a Gnuplot reference instead.
   */
  template <typename T>
    void plot_images_greyscale_4dim(const Matrix<T>& images, std::string title, std::vector<std::string> options = {}) {
    Gnuplot gp;
    plot_images_greyscale_4dim(gp, images, title, options);
  }


  /*
   * Plot the images in the supplied "images" matrix.
   *
   * The supplied "images" matrix must have order 3 where the last 2 dimensions are image height and
   * image width. That is, the allowable size for "images" is:
   *
   * (dim0, height, width)
   * in which case there are dim0 images in total.
   *
   * An example usage of this function is when "images" corresponds to the hidden activations which are
   * the output of a convolutional layer in a neural
   * network. In this case, dim0 is the "image depth" or "channel count".
   *
   * Since there is no RGB channel dimension, the images are assumed to be greyscale of size (height, width).
   *
   * The images are arranged into a rectangular grid of images where each image has size (height, width).
   *
   * This version requires that the user get a Gnuplot instance first before calling this
   * function. This version should be used if you would like to be able to keep redrawing
   * the plot in the same figure window as new or different data becomes available, or if you would like like to use
   * the "multiplot" feature to put multiple plots into the same window.
   */
  template <typename T>
    void plot_images_greyscale_3dim(Gnuplot& gp, const Matrix<T>& images, std::string title, std::vector<std::string> options = {}) {
    if (images.order() != 3) {
      std::cerr << "plot_images_greyscale_3dim(): supplied images matrix is wrong order." << std::endl;
      exit(1);
    }
    const int image_count = images.extent(0); // dim0
    const int height = images.extent(1);
    const int width = images.extent(2);
    const int plot_rows = std::sqrt(image_count);
    const int plot_cols = std::ceil(static_cast<float>(image_count)/static_cast<float>(plot_rows));
    //std::cout << "plot_rows = " << plot_rows << std::endl;
    //std::cout << "plot_cols = " << plot_cols << std::endl;
    //std::cout << "height = " << height << std::endl;
    //std::cout << "width = " << width << std::endl;
    const int border_thickness = 1; // Thickness of border between images.
    const int out_image_rows = plot_rows*height + plot_rows*border_thickness;
    const int out_image_cols = plot_cols*width + plot_cols*border_thickness;
    //std::cout << "out_image_rows = " << out_image_rows << std::endl;
    //std::cout << "out_image_cols = " << out_image_cols << std::endl;
    Matrix<T> out_image(out_image_rows, out_image_cols);

    // We want the border to appear black (or same color as the minimum value).
    //float minval = min_value(images);
    //std::cout << "minval = " << minval << std::endl;
    //set_value(out_image, minval);
    // Or set the border to be white (or same color as max value)
    float maxval = max_value(images);
    set_value(out_image, maxval);

    int cur_image = 0;
    for (int p_row = 0; p_row != plot_rows; ++p_row) {
      for (int p_col = 0; p_col != plot_cols; ++p_col) {
        // Calculate current offset from the (0,0) corner of the output plot image.
        int row_offset = p_row*(height + border_thickness);
        int col_offset = p_col*(width + border_thickness);
        if (cur_image < image_count) {
          for (int i = 0; i != height; ++i) {
            for (int j = 0; j != width; ++j) {
              out_image(row_offset + i, col_offset + j) = images(cur_image, i,j);
            }
          }
        }
        ++cur_image;
      }
    }
    // Want to keep aspect ratio so that pixels remain square-shaped:
    float ratio = static_cast<float>(plot_rows)/static_cast<float>(plot_cols);
    std::ostringstream cmd_str;
    cmd_str << "set size ratio " << ratio;
    options.push_back(cmd_str.str());
    //std::cout << "ratio command: " << cmd_str.str() << std::endl;
    // Now plot it.
    plot_image_greyscale(gp, out_image, title, options);


  }

  /*
   * Plot the images in the supplied "images" matrix.
   *
   * The supplied "images" matrix must have order 3 where the last 2 dimensions are image height and
   * image width. That is, the allowable size for "images" is:
   *
   * (dim0, height, width)
   * in which case there are dim0 images in total.
   *
   * An example usage of this function is when "images" corresponds to the hidden activations which are
   * the output of a convolutional layer in a neural
   * network. In this case, dim0 is the "image depth" or "channel count".
   *
   * Since there is no RGB channel dimension, the images are assumed to be greyscale of size (height, width).
   *
   * The images are arranged into a rectangular grid of images where each image has size (height, width).
   *
   *
   */
  template <typename T>
    void plot_images_greyscale_3dim(const Matrix<T>& images, std::string title, std::vector<std::string> options = {}) {
    Gnuplot gp;
    plot_images_greyscale_3dim(gp, images, title, options);
  }


  /*
   * Plot the images in the supplied "images" matrix.
   *
   * The supplied "images" matrix must have order 3 or 4 where the last 2 dimensions are image height and
   * image width. That is, the allowable sizes for "images" are:
   *
   * (dim0, dim1, height, width)
   * in which case there are dim0*dim1 images in total
   * or
   * (dim0, height, width)
   * in which case there are dim0 images in total.
   *
   * Since there is no RGB channel dimension, the images are assumed to be greyscale of size (height, width).
   *
   * The images are arranged into an approximately square grid and plotted.
   *
   */
  // fixme
  template <typename T>
    void plot_images_rgb(const Matrix<T>& images, std::string title, std::vector<std::string> options = {}, bool normalize = true) {
    int image_count = 0;
    int height = 0;
    int width = 0;
    if (images.order() == 4) {
      image_count = images.extent(0)*images.extent(1);
      height = images.extent(2);
      width = images.extent(3);
    } else if (images.order() == 3) {
      image_count = images.extent(0);
      height = images.extent(1);
      width = images.extent(2);
    } else {
      std::cerr << "plot_images_greyscale(): supplied images matrix is wrong order." << std::endl;
      exit(1);
    }
    const int plot_rows = std::sqrt(image_count);
    const int plot_cols = std::ceil(static_cast<float>(image_count)/static_cast<float>(plot_rows));
    //std::cout << "plot_rows = " << plot_rows << std::endl;
    //std::cout << "plot_cols = " << plot_cols << std::endl;
    const int border_thickness = 1; // Thickness of border between images.
    const int out_image_rows = plot_rows*height + plot_rows*border_thickness;
    const int out_image_cols = plot_cols*width + plot_cols*border_thickness;
    Matrix<T> out_image(out_image_rows, out_image_cols);

    int cur_image = 0;
    for (int p_col = 0; p_col != plot_cols; ++p_col) {
      for (int p_row = 0; p_row != plot_rows; ++p_row) {
        // Calculate current offset from the (0,0) corner of the output plot image.
        int row_offset = p_row*(height + border_thickness);
        int col_offset = p_col*(width + border_thickness);

        if (images.order() == 4) {

        } else if (images.order() == 3) {

        }

      }
    }

  }


}

#endif  /* _PLOTUTILITIES_H */
