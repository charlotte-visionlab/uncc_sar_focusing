/* 
 * File:   mxComplexSingleClass.hpp
 * Author: arwillis
 *
 * Created on June 16, 2021, 12:14 PM
 */
#ifndef MXCOMPLEXSINGLECLASS_HPP
#define MXCOMPLEXSINGLECLASS_HPP

class mxComplexSingleClass;
#ifndef NO_MATLAB

#include <mex.h>    // Matlab library includes
#include <matrix.h> // Matlab mxComplexSingle struct

typedef mxComplexSingleClass Complex;
#define polarToComplex mxComplexSingleClass::polar
#define conjugateComplex mxComplexSingleClass::conj

#else

typedef float mxSingle;

typedef struct {
    mxSingle real, imag;
} mxComplexSingle;

//#include <complex>
//typedef std::complex<float> Complex;
//#define polarToComplex std::polar
//#define conjugateComplex std::conj

typedef mxComplexSingleClass Complex;
#define polarToComplex mxComplexSingleClass::polar
#define conjugateComplex mxComplexSingleClass::conj

#endif

// TODO: We should be extending mxComplexSingle with an ExpressionTemplate class
//       then extending the ExpressionTemplate class with the mxComplexSingleClass
// Reference and explanation of Expression Templates:
//  https://en.wikipedia.org/wiki/Expression_templates
// Why implementing them may be valuable for performance:
//  https://stackoverflow.com/questions/6850807/why-is-valarray-so-slow
// Note: Boost's multi-precision complex handles expression templates
//  https://www.boost.org/doc/libs/develop/libs/multiprecision/doc/html/boost_multiprecision/tut/complex/cpp_complex.html
// std::comples implementation
//  https://code.woboq.org/gcc/libstdc++-v3/include/std/complex.html

class mxComplexSingleClass : public mxComplexSingle {
public:

    mxComplexSingleClass() {
    }

    mxComplexSingleClass(mxComplexSingle& x) {
        //std::cout << "here1 " << x.real << std::endl;
        real = x.real;
        imag = x.imag;
    }

    mxComplexSingleClass(const float& _real, const float& _imag) {
        //std::cout << "here2 " << _real << std::endl;
        real = _real;
        imag = _imag;
    }

    // needed for valarray operations like x /= x.size() 
    // when x is a std::valarray<mxComplexSingleClass>    

    template <typename _Tp>
    inline mxComplexSingleClass(const _Tp& _real) {
        real = _real;
        imag = 0.0f;
    }

    inline static mxComplexSingleClass conj(const mxComplexSingleClass& x) {
        mxComplexSingleClass z;
        z.real = x.real;
        z.imag = -x.imag;
        return z;
    }

    inline static mxComplexSingleClass polar(mxComplexSingleClass& x) {
        mxComplexSingleClass z;
        z.real = norm(x);
        z.imag = std::atan2(x.imag, x.real);
        return z;
    }

    // TODO: functions log(), cos(), sin(), tan(), sqrt() not implemented

    template <typename _Tp>
    inline static mxComplexSingleClass polar(const _Tp& r, const _Tp& phi) {
        return mxComplexSingleClass(r * std::cos(phi), r * std::sin(phi));
    }

    inline static float norm(const mxComplexSingleClass& z) {
        const float __x = z.real;
        const float __y = z.imag;
        return __x * __x + __y * __y;
    }

    inline static float abs(const mxComplexSingleClass& z) {
        float __x = z.real;
        float __y = z.imag;
        const float __s = std::max(std::abs(__x), std::abs(__y));
        if (__s == 0.0f)
            return __s;
        __x /= __s;
        __y /= __s;
        return __s * std::sqrt(__x * __x + __y * __y);
    }

    template <typename _Tp>
    inline mxComplexSingleClass operator/(const _Tp& rhs) {
        mxComplexSingleClass z = *this;
        z /= rhs;
        return z;
    }

    inline mxComplexSingleClass operator*(const mxComplexSingleClass& rhs) {
        mxComplexSingleClass z = *this;
        z *= rhs;
        return z;
    }

    template <typename _Tp>
    inline mxComplexSingleClass operator*(const _Tp& rhs) {
        mxComplexSingleClass z = *this;
        z *= rhs;
        return z;
    }

    inline mxComplexSingleClass operator+(const mxComplexSingleClass& rhs) {
        mxComplexSingleClass z = *this;
        z += rhs;
        return z;
    }

    inline mxComplexSingleClass operator-(const mxComplexSingleClass& rhs) {
        mxComplexSingleClass z = *this;
        z -= rhs;
        return z;
    }

    inline mxComplexSingleClass operator=(const mxComplexSingle& other) {
        real = other.real;
        imag = other.imag;
        return *this;
    }

    template <typename _Tp>
    inline mxComplexSingleClass operator=(const _Tp& other) {
        real = other;
        imag = 0;
        return *this;
    }

    inline mxComplexSingleClass& operator=(const mxComplexSingleClass& other) {
        real = other.real;
        imag = other.imag;
        return *this;
    }

    inline mxComplexSingleClass& operator+=(const mxComplexSingleClass& rhs) {
        real += rhs.real;
        imag += rhs.imag;
        return *this;
    }

    template <typename _Tp>
    inline mxComplexSingleClass& operator+=(const _Tp& rhs) {
        real += rhs;
        return *this;
    }

    template <typename _Tp>
    inline mxComplexSingleClass& operator-=(const _Tp& rhs) {
        real -= rhs;
        return *this;
    }

    inline mxComplexSingleClass& operator-=(const mxComplexSingleClass& rhs) {
        real -= rhs.real;
        imag -= rhs.imag;
        return *this;
    }

    template <typename _Tp>
    inline mxComplexSingleClass& operator*=(const _Tp& rhs) {
        imag *= rhs;
        real *= rhs;
        return *this;
    }

    inline mxComplexSingleClass& operator*=(const mxComplexSingleClass& rhs) {
        const float __r = real * rhs.real - imag * rhs.imag;
        imag = real * rhs.imag + imag * rhs.real;
        real = __r;
        return *this;
    }

    inline mxComplexSingleClass& operator/=(const mxComplexSingleClass& rhs) {
        //std::cout << "this=" << *this << " rhs=" << rhs << std::endl;
        const float __r = real * rhs.real + imag * rhs.imag;
        const float __n = norm(rhs);
        imag = (imag * rhs.real - real * rhs.imag) / __n;
        real = __r / __n;
        return *this;
    }

    template <typename _Tp>
    inline mxComplexSingleClass& operator/=(const _Tp& rhs) {
        real /= rhs;
        imag /= rhs;
        return *this;
    }

    //friend _GLIBCXX_CONSTEXPR bool operator==(const mxComplexSingleClass& __x, const mxComplexSingleClass& __y);
    friend std::ostream& operator<<(std::ostream& output, const mxComplexSingleClass &c);
    friend std::istream& operator>>(std::istream& input, const mxComplexSingleClass& c);
};

template<typename _Tp>
inline mxComplexSingleClass operator+(const mxComplexSingleClass& __x, const mxComplexSingleClass& __y) {
    mxComplexSingleClass __r = __x;
    __r += __y;
    return __r;
}

//template<typename _Tp>
//inline mxComplexSingleClass operator+(const mxComplexSingleClass& __x, const _Tp& __y) {
//    mxComplexSingleClass __r = __x;
//    __r += __y;
//    return __r;
//}

template<typename _Tp>
inline mxComplexSingleClass operator+(const _Tp& __x, const mxComplexSingleClass& __y) {
    mxComplexSingleClass __r = __y;
    __r += __x;
    return __r;
}
//@}
//@{
///  Return new complex value @a x minus @a y.

template<typename _Tp>
inline mxComplexSingleClass operator-(const mxComplexSingleClass& __x, const mxComplexSingleClass& __y) {
    mxComplexSingleClass __r = __x;
    __r -= __y;
    return __r;
}

template<typename _Tp>
inline mxComplexSingleClass operator-(const mxComplexSingleClass& __x, const _Tp& __y) {
    mxComplexSingleClass __r = __x;
    __r -= __y;
    return __r;
}

template<typename _Tp>
inline mxComplexSingleClass operator-(const _Tp& __x, const mxComplexSingleClass& __y) {
    mxComplexSingleClass __r(__x, -__y.imag());
    __r -= __y.real();
    return __r;
}

template<typename _Tp>
inline mxComplexSingleClass operator*(const mxComplexSingleClass& __x, const mxComplexSingleClass& __y) {
    mxComplexSingleClass __r = __x;
    __r *= __y;
    return __r;
}

template<typename _Tp>
inline mxComplexSingleClass operator*(const mxComplexSingleClass& __x, const _Tp& __y) {
    mxComplexSingleClass __r = __x;
    __r *= __y;
    return __r;
}

template<typename _Tp>
inline mxComplexSingleClass operator*(const _Tp& __x, const mxComplexSingleClass& __y) {
    mxComplexSingleClass __r = __y;
    __r *= __x;
    return __r;
}

template<typename _Tp>
inline mxComplexSingleClass operator/(const mxComplexSingleClass& __x, const mxComplexSingleClass& __y) {
    mxComplexSingleClass __r = __x;
    __r /= __y;
    return __r;
}

template<typename _Tp>
inline mxComplexSingleClass operator/(const mxComplexSingleClass& __x, const _Tp& __y) {
    mxComplexSingleClass __r = __x;
    __r /= __y;
    return __r;
}

template<typename _Tp>
inline mxComplexSingleClass operator/(const _Tp& __x, const mxComplexSingleClass& __y) {
    mxComplexSingleClass __r = __x;
    __r /= __y;
    return __r;
}

// Templates for constant expressions with ==

inline _GLIBCXX_CONSTEXPR bool operator==(const mxComplexSingleClass& __x, const mxComplexSingleClass& __y) {
    return __x.real == __y.real && __x.imag == __y.imag;
}

template<typename _Tp>
inline _GLIBCXX_CONSTEXPR bool operator==(const mxComplexSingleClass& __x, const _Tp& __y) {
    return __x.real == __y && __x.imag == _Tp();
}

template<typename _Tp>
inline _GLIBCXX_CONSTEXPR bool operator==(const _Tp& __x, const mxComplexSingleClass& __y) {
    return __x == __y.real && _Tp() == __y.imag;
}

// Templates for constant expressions with !=

//template<typename _Tp>
//inline _GLIBCXX_CONSTEXPR bool operator!=(const mxComplexSingleClass& __x, const mxComplexSingleClass& __y) {
//    return __x.real != __y.real || __x.imag != __y.imag;
//}
//
//template<typename _Tp>
//inline _GLIBCXX_CONSTEXPR bool operator!=(const mxComplexSingleClass& __x, const _Tp& __y) {
//    return __x.real != __y || __x.imag != _Tp();
//}
//template<typename _Tp>
//inline _GLIBCXX_CONSTEXPR bool operator!=(const _Tp& __x, const mxComplexSingleClass& __y) {
//    return __x != __y.real || _Tp() != __y.imag;
//}

// std::cout << "Enter a complex number (a+bi) : " << std::endl;
// std::cin >> x;

inline std::istream& operator>>(std::istream& input, mxComplexSingleClass& __x) {
    char plus;
    char i;
    input >> __x.real >> plus >> __x.imag >> i;
    return input;
}
///  Insertion operator for complex values.

inline std::ostream& operator<<(std::ostream& output, const mxComplexSingleClass& __x) {
    output << __x.real << ((__x.imag < 0) ? "" : "+") << __x.imag << "i";
    return output;
}

#endif /* MXCOMPLEXSINGLECLASS_HPP */

