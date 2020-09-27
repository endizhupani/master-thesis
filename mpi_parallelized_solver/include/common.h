/**
 * common.h 
 * Defines common data structures
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
#include "mpi.h"
#include <string>
#include <iostream>
#ifndef COMMON_H
#define COMMON_H
/**
 * @brief The type of the partition neighbour
 * 
 */
enum PartitionNeighbourType
{
    LEFT_NEIGHTBOUR,
    RIGHT_NEIGHBOUR,
    BOTTOM_NEIGHBOUR,
    TOP_NEIGHBOUR
};

/**
 * @brief The neighbour of the partition. Identified by the processor number and the position relative to the target
 * 
 */
struct PartitionNeighbour
{
public:
    // The id of the processor that holds this neighbour
    int id;
    // Defines the type which determines the position of the neighbor
    enum PartitionNeighbourType type;

    std::string GetNeighborId()
    {
        if (id == MPI_PROC_NULL)
        {
            return "NULL";
        }

        return std::to_string(id);
    }
};

#endif // !COMMON_H
