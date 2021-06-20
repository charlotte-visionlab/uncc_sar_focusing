/* 
 * File:   mxComplexSingleClass.hpp
 * Author: arwillis
 *
 * Created on June 16, 2021, 12:14 PM
 */
#ifndef MXCOMPLEXSINGLECLASS_HPP
#define MXCOMPLEXSINGLECLASS_HPP

#ifndef NO_MATLAB

#include <matrix.h> // Matlab mxComplexSingle struct

#else

typedef float mxSingle;
struct mxComplexSingle {
    mxSingle real, imag;
};

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

template<typename __Tp>
class mxComplexSingleClass {
public:
    __Tp real;
    __Tp imag;

    mxComplexSingleClass(mxComplexSingle& x) : real(x.real), imag(x.imag) {
        //std::cout << "here1 " << x.real << std::endl;
    }

    //template<typename __Tp>
    mxComplexSingleClass(const __Tp& _real = __Tp(), const __Tp& _imag = __Tp()) :
    real(_real), imag(_imag) {
        //std::cout << "here2 " << _real << std::endl;
    }

    template<typename __oTp>
    mxComplexSingleClass(const mxComplexSingleClass<__oTp> c) : real(c.real), imag(c.imag) {

    }
    // needed for valarray operations like x /= x.size() 
    // when x is a std::valarray<mxComplexSingleClass>    

    template <typename __oTp>
    inline mxComplexSingleClass(const __oTp& _real) {
        real = _real;
        imag = 0;
    }

    inline static mxComplexSingleClass<__Tp> conj(const mxComplexSingleClass<__Tp>& x) {
        mxComplexSingleClass<__Tp> z;
        z.real = x.real;
        z.imag = -x.imag;
        return z;
    }

    template <typename __oTp>
    inline static mxComplexSingleClass<__Tp> polar(mxComplexSingleClass<__oTp>& x) {
        mxComplexSingleClass<__Tp> z;
        z.real = norm(x);
        z.imag = std::atan2(x.imag, x.real);
        return z;
    }

    // TODO: functions log(), cos(), sin(), tan(), sqrt() not implemented

    template <typename __oTp1, typename __oTp2>
    inline static mxComplexSingleClass<__Tp> polar(const __oTp1& r, const __oTp2& phi) {
        return mxComplexSingleClass<__Tp>(r * std::cos(phi), r * std::sin(phi));
    }

    inline static __Tp norm(const mxComplexSingleClass<__Tp>& z) {
        const __Tp __x = z.real;
        const __Tp __y = z.imag;
        return __x * __x + __y * __y;
    }

    inline static __Tp abs(const mxComplexSingleClass<__Tp>& z) {
        __Tp __x = z.real;
        __Tp __y = z.imag;
        const __Tp __s = std::max(std::abs(__x), std::abs(__y));
        if (__s == 0.0f)
            return __s;
        __x /= __s;
        __y /= __s;
        return __s * std::sqrt(__x * __x + __y * __y);
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp> operator/(const __oTp& rhs) {
        mxComplexSingleClass<__Tp> z = *this;
        z /= rhs;
        return z;
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp> operator*(const mxComplexSingleClass<__oTp>& rhs) {
        mxComplexSingleClass<__Tp> z = *this;
        z *= rhs;
        return z;
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp> operator*(const __oTp& rhs) {
        mxComplexSingleClass<__Tp> z = *this;
        z *= rhs;
        return z;
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp> operator+(const mxComplexSingleClass<__oTp>& rhs) {
        mxComplexSingleClass<__Tp> z = *this;
        z += rhs;
        return z;
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp> operator-(const mxComplexSingleClass<__oTp>& rhs) {
        mxComplexSingleClass<__Tp> z = *this;
        z -= rhs;
        return z;
    }

    inline mxComplexSingleClass<__Tp> operator=(const mxComplexSingle& other) {
        real = other.real;
        imag = other.imag;
        return *this;
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp> operator=(const __oTp& other) {
        real = other;
        imag = 0;
        return *this;
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp>& operator=(const mxComplexSingleClass& other) {
        real = other.real;
        imag = other.imag;
        return *this;
    }

    inline mxComplexSingleClass<__Tp>& operator+=(const mxComplexSingleClass& rhs) {
        real += rhs.real;
        imag += rhs.imag;
        return *this;
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp>& operator+=(const __oTp& rhs) {
        real += rhs;
        return *this;
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp>& operator-=(const __oTp& rhs) {
        real -= rhs;
        return *this;
    }

    //template <typename __Tp>
    inline mxComplexSingleClass<__Tp>& operator-=(const mxComplexSingleClass<__Tp>& rhs) {
        real -= rhs.real;
        imag -= rhs.imag;
        return *this;
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp>& operator*=(const __oTp& rhs) {
        imag *= rhs;
        real *= rhs;
        return *this;
    }

    inline mxComplexSingleClass<__Tp>& operator*=(const mxComplexSingleClass<__Tp>& rhs) {
        const float __r = real * rhs.real - imag * rhs.imag;
        imag = real * rhs.imag + imag * rhs.real;
        real = __r;
        return *this;
    }

    inline mxComplexSingleClass<__Tp>& operator/=(const mxComplexSingleClass<__Tp>& rhs) {
        //std::cout << "this=" << *this << " rhs=" << rhs << std::endl;
        const __Tp __r = real * rhs.real + imag * rhs.imag;
        const __Tp __n = norm(rhs);
        imag = (imag * rhs.real - real * rhs.imag) / __n;
        real = __r / __n;
        return *this;
    }

    template <typename __oTp>
    inline mxComplexSingleClass<__Tp>& operator/=(const __oTp& rhs) {
        real /= rhs;
        imag /= rhs;
        return *this;
    }

    //friend _GLIBCXX_CONSTEXPR bool operator==(const mxComplexSingleClass& __x, const mxComplexSingleClass& __y);
    template <typename __oTp>
    friend std::ostream& operator<<(std::ostream& output, const mxComplexSingleClass<__oTp>& c);
    template <typename __oTp>
    friend std::istream& operator>>(std::istream& input, const mxComplexSingleClass<__oTp>& c);
};

template<typename __Tp>
inline mxComplexSingleClass<__Tp> operator+(const mxComplexSingleClass<__Tp>& __x, const mxComplexSingleClass<__Tp>& __y) {
    mxComplexSingleClass<__Tp> __r = __x;
    __r += __y;
    return __r;
}

template<typename __Tp>
inline mxComplexSingleClass<__Tp> operator+(const mxComplexSingleClass<__Tp>& __x, const __Tp& __y) {
    mxComplexSingleClass<__Tp> __r = __x;
    __r += __y;
    return __r;
}

template<typename __Tp>
inline mxComplexSingleClass<__Tp> operator+(const __Tp& __x, const mxComplexSingleClass<__Tp>& __y) {
    mxComplexSingleClass<__Tp> __r = __y;
    __r += __x;
    return __r;
}
//@}
//@{
///  Return new complex value @a x minus @a y.

template<typename __Tp>
inline mxComplexSingleClass<__Tp> operator-(const mxComplexSingleClass<__Tp>& __x, const mxComplexSingleClass<__Tp>& __y) {
    mxComplexSingleClass<__Tp> __r = __x;
    __r -= __y;
    return __r;
}

template<typename __Tp>
inline mxComplexSingleClass<__Tp> operator-(const mxComplexSingleClass<__Tp>& __x, const __Tp& __y) {
    mxComplexSingleClass<__Tp> __r = __x;
    __r -= __y;
    return __r;
}

template<typename __Tp, typename __oTp>
inline mxComplexSingleClass<__Tp> operator-(const __oTp& __x, const mxComplexSingleClass<__Tp>& __y) {
    mxComplexSingleClass<__Tp> __r(__x, -__y.imag);
    __r -= __y.real;
    return __r;
}

template<typename __Tp, typename __oTp>
inline mxComplexSingleClass<__Tp> operator*(const mxComplexSingleClass<__Tp>& __x, const mxComplexSingleClass<__oTp>& __y) {
    mxComplexSingleClass<__Tp> __r = __x;
    __r *= __y;
    return __r;
}

template<typename __Tp>
inline mxComplexSingleClass<__Tp> operator*(const mxComplexSingleClass<__Tp> & __x, const __Tp& __y) {
    mxComplexSingleClass<__Tp>  __r = __x;
    __r *= __y;
    return __r;
}

template<typename __Tp>
inline mxComplexSingleClass<__Tp> operator*(const __Tp& __x, const mxComplexSingleClass<__Tp>& __y) {
    mxComplexSingleClass<__Tp> __r = __y;
    __r *= __x;
    return __r;
}

template<typename __Tp>
inline mxComplexSingleClass<__Tp> operator/(const mxComplexSingleClass<__Tp>& __x, const mxComplexSingleClass<__Tp>& __y) {
    mxComplexSingleClass<__Tp> __r = __x;
    __r /= __y;
    return __r;
}

template<typename __Tp>
inline mxComplexSingleClass<__Tp> operator/(const mxComplexSingleClass<__Tp>& __x, const __Tp& __y) {
    mxComplexSingleClass<__Tp> __r = __x;
    __r /= __y;
    return __r;
}

template<typename __Tp>
inline mxComplexSingleClass<__Tp> operator/(const __Tp& __x, const mxComplexSingleClass<__Tp>& __y) {
    mxComplexSingleClass<__Tp> __r = __x;
    __r /= __y;
    return __r;
}

// Templates for constant expressions with ==

template<typename __Tp>
inline _GLIBCXX_CONSTEXPR bool operator==(const mxComplexSingleClass<__Tp>& __x, const mxComplexSingleClass<__Tp>& __y) {
    return __x.real == __y.real && __x.imag == __y.imag;
}

template<typename __Tp>
inline _GLIBCXX_CONSTEXPR bool operator==(const mxComplexSingleClass<__Tp>& __x, const __Tp& __y) {
    return __x.real == __y && __x.imag == __Tp();
}

template<typename __Tp>
inline _GLIBCXX_CONSTEXPR bool operator==(const __Tp& __x, const mxComplexSingleClass<__Tp>& __y) {
    return __x == __y.real && __Tp() == __y.imag;
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

template <typename __Tp>
inline std::istream& operator>>(std::istream& input, mxComplexSingleClass<__Tp>& __x) {
    char plus;
    char i;
    input >> __x.real >> plus >> __x.imag >> i;
    return input;
}
///  Insertion operator for complex values.

template <typename __Tp>
inline std::ostream& operator<<(std::ostream& output, const mxComplexSingleClass<__Tp>& __x) {
    output << __x.real << ((__x.imag < 0) ? "" : "+") << __x.imag << "i";
    return output;
}

#endif /* MXCOMPLEXSINGLECLASS_HPP */

