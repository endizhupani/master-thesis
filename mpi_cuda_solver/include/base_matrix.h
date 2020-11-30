/**
 * base_matrix.h
 * 
 * ------------------------------------------------------
 * Copyright (c) 2020 Endi Zhupani
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without restriction, 
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
 * is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or 
 * substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef BASE_MATRIX_H
#define BASE_MATRIX_H
/**
 * @brief Common data objects and behavior
 * 
 */

namespace pde_solver
{
    /**
     * @brief Base matrix class with common behavior used by all types of 2D matrix configurations.
     * 
     */
    class BaseMatrix
    {
    protected:
        // global matrix width
        int matrix_width_;

        // global matrix height
        int matrix_height_;
        // array that holds the left border of the partition
        // float *left_border_;

        // // array that holds the right border of the partition
        // float *right_border_;

        // array holding the inner points of the matrix as well as the top and bottom border
        float *inner_points_;

    public:
        /**
     * @brief Construct a new Base Matrix object
     * 
     * @param width width of the matrix
     * @param height height of the matrix
     */
        BaseMatrix(int width, int height);

        /**
         * @brief Destroy the Base Matrix object
         * 
         */
        ~BaseMatrix();

        /**
         * @brief Initializes the matrix with the same value on all elements
         * 
         * @param value Value to assign to the elements of the matrix
         */
        void Init(float value);

        /**
         * @brief Initializes the matrix with specific values on borders and another value for all the inner elements.
         * 
         * @param inner_value Value to be assigned to non-bordering elements of the global matrix
         * @param left_border_value Value to be assigned to the left border of the global matrix
         * @param right_border_value Value to be assigned to the right border of the global matrix
         * @param bottom_border_value Value to be assigned to the bottom border of the global matrix
         * @param top_border_value Value to be assigned to the top border of the global matrix
         */
        void Init(float inner_value, float left_border_value, float right_border_value, float bottom_border_value, float top_border_value);

        /**
         * @brief gets the matrix contents
         * 
         * @return float* the matrix stored as an array
         */
        virtual void ShowMatrix() = 0;
    };
} // namespace pde_solver

#endif // !BASE_MATRIX_H
