/*
 * Copyright (C) 2021 Andrew R. Willis
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


/* 
 * File:   uncc_simplematrix.hpp
 * Author: arwillis
 *
 * Created on April 24, 2022, 7:20 AM
 */

#ifndef UNCC_SIMPLEMATRIX_HPP
#define UNCC_SIMPLEMATRIX_HPP

#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

class Range {
public:
    int start, end, stride;

    Range(int _end) : Range(0, _end, 1) {
    }

    Range(int _start, int _end) : Range(_start, _end, 1) {
    }

    Range(int _start, int _end, int _stride) : start(_start), end(_end), stride(_stride) {
        if (start < 0) {
            std::cout << "Error on  Range construction. Start < 0." << std::endl;
            start = 0;
        }
        if (end < 0) {
            std::cout << "Error on  Range construction. End < 0." << std::endl;
            end = 0;
        }
        if (stride == 0) {
            std::cout << "Error on  Range construction. Stride = 0." << std::endl;
            stride = (start < end) ? 1 : -1;
        }
        if (start > end && stride > 0) {
            std::cout << "Error on  Range construction. Stride has incorrect sign." << std::endl;
            stride = -1;
        } else if (start < end && stride < 0) {
            std::cout << "Error on  Range construction. Stride has incorrect sign." << std::endl;
            stride = 1;
        }
        int nelem = (end - start) / stride;
        if (nelem == 0) {
            std::cout << "Error on  Range construction. Number of elements in range is 0." << std::endl;
        }
    }

    std::vector<int> values() {
        std::vector<int> values;
        int nelem = (end - start) / stride;
        for (int elemIdx = 0; elemIdx < nelem; elemIdx++) {
            values.push_back(start + elemIdx * stride);
        }
        return values;
    }

    virtual ~Range() {
    }
};

template<typename __numTp>
struct SimpleMatrix {
public:
    std::vector<int> shape;
    std::vector<__numTp> data;
    typedef std::shared_ptr<SimpleMatrix> Ptr;

    int nelem() {
        return (shape.size() == 0) ? 0 :
                std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>()); // / (polarityIdx == -1 ? 1 : shape[polarityIdx]);
    }

    bool isEmpty() {
        return shape.size() == 0;
    }

    int sub2ind(std::vector<int> ivec) {
        return sub2ind(ivec, ivec.size() - 1);
    }

    __numTp at(std::vector<int> idxs) {
        return data[sub2ind(idxs)];
    }

private:
    // Range-based restructuring of SAR aperture data uses recursive 
    // functions for multi-dimensional data indexing. This is slow. I have
    // unrolled some functions for common invocations to replace purely 
    // recursive functions (similar to what would happen 
    // via #pragma inline_recursion(on) for a MSVC compiler).
    // https://docs.microsoft.com/en-us/cpp/preprocessor/inline-recursion?view=msvc-160
    //
    // Ultimately a better and more intelligent way to perform range-based
    // restructuring of memory is required. This will have to do in the interim.
    //

    inline
    int nelem(int dim) {
        return (shape.size() == 0) ? 0 :
                std::accumulate(std::begin(shape), std::begin(shape) + dim + 1, 1, std::multiplies<int>());
    }

    int nelem_slow(int dim) {
        return (dim == 0) ? shape[0] : shape[dim] * nelem(dim - 1);
    }

    inline
    int sub2ind(std::vector<int> ivec, int idx) {
        int retval = 0;
        if (retval > 5) {
            return sub2ind_slow(ivec, idx);
        }
        switch (idx) {
            case 5:
                retval += ivec[5];
                retval *= shape[4];
            case 4:
                retval += ivec[4];
                retval *= shape[3];
            case 3:
                retval += ivec[3];
                retval *= shape[2];
            case 2:
                retval += ivec[2];
                retval *= shape[1];
            case 1:
                retval += ivec[1];
                retval *= shape[0];
            case 0:
                retval += ivec[0];
        }
        return retval;
    }

    int sub2ind_slow(std::vector<int> ivec, int idx) {
        if (idx == 0)
            return ivec[0];
        else
            return ivec[idx] * nelem(idx - 1) + sub2ind(ivec, idx - 1);
    }

    void iterate(std::vector<std::vector<int>> dimIdxs, std::vector<int>& cpos, int dim, std::vector<__numTp>& out) {
        std::vector<int>::const_iterator itr;
        for (itr = dimIdxs[dim].begin(); itr != dimIdxs[dim].end(); ++itr) {
            // do stuff  breadth-first
            cpos[dim] = *itr;
            if (dim > 0) {
                iterate(dimIdxs, cpos, dim - 1, out);
            }
            if (dim == 0) {
                out.push_back(this->at(cpos));
                //std::cout << ".at( ";
                //for (std::vector<int>::const_iterator i = cpos.begin(); i != cpos.end(); ++i) {
                //    std::cout << *i << ", ";
                //}
                //std::cout << " )" << std::endl;
            }
            // do stuff  depth-first
        }
    }

public:
    // converts the matrix to a vector using column-major order

    template <typename __dstTp>
    std::vector<__dstTp> toVector() {
        int numElems = nelem();
        std::vector<__dstTp> vecdata(numElems);
        for (int idx = 0; idx < numElems; idx++) {
            vecdata[idx] = (__dstTp) data[idx];
        }
        return vecdata;
    }   

    std::vector<__numTp> getData(std::vector<std::vector<int>> idxs) {
        std::vector<__numTp> data;
        std::vector<int> b(idxs.size(), 0);
        iterate(idxs, b, idxs.size() - 1, data);
        return data;
    }

    SimpleMatrix getRange(std::vector<Range> rangeSelection) {
        SimpleMatrix<__numTp> output;
        std::vector<std::vector<int>> idxs;
        for (Range r : rangeSelection) {
            std::vector<int> range_indices = r.values();
            if (range_indices.size() > 1) {
                output.shape.push_back(range_indices.size());
            }
            idxs.push_back(r.values());
        }
        output.data = getData(idxs);
        return output;
    }

    // TODO: Add zero-padding code for the SimpleMatrix class

    void pad(std::vector<int> padding, __numTp value) {

    }

    SimpleMatrix::Ptr create() {
        return std::make_shared<SimpleMatrix>();
    }

    template <typename _Tp>
    friend std::ostream& operator<<(std::ostream& output, const SimpleMatrix<__numTp> &c);

};

template <typename __numTp>
inline std::ostream& operator<<(std::ostream& output, const SimpleMatrix<__numTp>& sMat) {
    int MAX_NUM_VARS_PRINTED = 10;
    std::string dimsStr;
    if (!sMat.shape.empty()) {
        dimsStr = std::accumulate(sMat.shape.begin() + 1, sMat.shape.end(),
                std::to_string(sMat.shape[0]), [](const std::string & a, int b) {
                    return a + ',' + std::to_string(b);
                });
    }
    output << "[" << dimsStr << "] = {"; \
    typename std::vector<__numTp>::const_iterator i;
    for (i = sMat.data.begin(); i != sMat.data.end() && i != sMat.data.begin() + MAX_NUM_VARS_PRINTED; ++i) {
        output << *i << ", ";
    }
    output << ((sMat.data.size() > MAX_NUM_VARS_PRINTED) ? " ..." : "") << " }";
    return output;
}

template<typename _src_realTp, typename _dst_realTp>
int import_MatrixReal(_src_realTp *data, int *dims, int ndims, SimpleMatrix<_dst_realTp>& sMat) {
    int totalsize = 1;
    for (int dimIdx = 0; dimIdx < ndims; dimIdx++) {
        sMat.shape.push_back(dims[dimIdx]);
        totalsize = totalsize * dims[dimIdx];
    }
    char *dp = (char *) data;
    sMat.data.insert(sMat.data.end(), (_src_realTp *) & data[0], (_src_realTp *) (dp + totalsize * sizeof (_src_realTp)));
    return EXIT_SUCCESS;
}

template<typename _src_numTp, typename _dst_numTp>
int import_Vector(_src_numTp *data, int *dims, int ndims, SimpleMatrix<_dst_numTp>& sMat) {
//    int totalsize = 1;
//    for (int dimIdx = 0; dimIdx < ndims; dimIdx++) {
//        sMat.shape.push_back(dims[dimIdx]);
//        totalsize = totalsize * dims[dimIdx];
//    }
//    char *dp = (char *) data;
//    sMat.data.insert(sMat.data.end(), (_src_numTp *) dp, (_src_numTp *) (dp + dims[0] * dims[1] * sizeof(_src_numTp)));
//    return EXIT_SUCCESS;
    return import_MatrixReal < _src_numTp, _dst_numTp>(data, dims, ndims, sMat);
}

template<typename _src_complexTp, typename _dst_complexTp>
int import_MatrixComplex(_src_complexTp* data, int *dims, int ndims, SimpleMatrix<_dst_complexTp>& sMat) {
    int totalsize = 1;
    for (int dimIdx = 0; dimIdx < ndims; dimIdx++) {
        sMat.shape.push_back(dims[dimIdx]);
        totalsize = totalsize * dims[dimIdx];
    }
    size_t stride = sizeof (_src_complexTp);
    char *dataptr = (char *) data;
    for (int idx = 0; idx < totalsize; idx++) {
        sMat.data.push_back(_dst_complexTp(*(_src_complexTp*) (dataptr + (2*idx) * stride), *(_src_complexTp*) (dataptr + ((2*idx)+1) * stride)));
    }
    return EXIT_SUCCESS;
}

#endif /* UNCC_SIMPLEMATRIX_HPP */

