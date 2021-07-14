/* 
 * File:   uncc_complex.hpp
 * Author: arwillis
 *
 * Created on June 16, 2021, 12:14 PM
 */
#ifndef UNCC_COMPLEX_HPP
#define UNCC_COMPLEX_HPP

// TODO: We should be extending uncc_complex with an ExpressionTemplate class
//       then extending the ExpressionTemplate class with the uncc_complex class
// Reference and explanation of Expression Templates:
//  https://en.wikipedia.org/wiki/Expression_templates
// Why implementing them may be valuable for performance:
//  https://stackoverflow.com/questions/6850807/why-is-valarray-so-slow
// Note: Boost's multi-precision complex handles expression templates
//  https://www.boost.org/doc/libs/develop/libs/multiprecision/doc/html/boost_multiprecision/tut/complex/cpp_complex.html
// std::complex implementation
//  https://code.woboq.org/gcc/libstdc++-v3/include/std/complex.html

template<typename __Tp>
class unccComplex {
public:
    __Tp _M_real;
    __Tp _M_imag;

    constexpr CUDAFUNCTION __Tp real() const {
        return _M_real;
    }

    constexpr CUDAFUNCTION __Tp imag() const {
        return _M_imag;
    }

    CUDAFUNCTION unccComplex(const __Tp& _real = __Tp(), const __Tp& _imag = __Tp()) :
    _M_real(_real), _M_imag(_imag) {
        //std::cout << "here2 " << _real << std::endl;
    }

    template<typename __oTp>
    CUDAFUNCTION unccComplex(const unccComplex<__oTp> c) : _M_real(c.real()), _M_imag(c.imag()) {

    }
    // needed for valarray operations like x /= x.size() 
    // when x is a std::valarray<mxComplexSingleClass>    

    template <typename __oTp>
    inline CUDAFUNCTION unccComplex(const __oTp& _real) {
        _M_real = __Tp(_real);
        _M_imag = 0;
    }

    inline static CUDAFUNCTION unccComplex<__Tp> conj(const unccComplex<__Tp>& x) {
        unccComplex<__Tp> z;
        z._M_real = x.real();
        z._M_imag = -x.imag();
        return z;
    }

    template <typename __oTp>
    inline static CUDAFUNCTION unccComplex<__Tp> polar(unccComplex<__oTp>& x) {
        unccComplex<__Tp> z;
        z._M_real = norm(x);
        z._M_imag = std::atan2(x.imag(), x.real());
        return z;
    }

    // TODO: functions log(), cos(), sin(), tan(), sqrt() not implemented

    template <typename __oTp1, typename __oTp2>
    inline static CUDAFUNCTION unccComplex<__Tp> polar(const __oTp1& r, const __oTp2& phi) {
        return unccComplex<__Tp>(r * std::cos(phi), r * std::sin(phi));
    }

    // Returns the square of the L2 norm of the complex number
    inline static __Tp norm(const unccComplex<__Tp>& z) {
        const __Tp __x = z.real();
        const __Tp __y = z.imag();
        return __x * __x + __y * __y;
    }

    inline static __Tp abs(const unccComplex<__Tp>& z) {
        __Tp __x = z.real();
        __Tp __y = z.imag();
        const __Tp __s = std::max(std::abs(__x), std::abs(__y));
        if (__s == 0.0f)
            return __s;
        __x /= __s;
        __y /= __s;
        return __s * std::sqrt(__x * __x + __y * __y);
    }

    template <typename __oTp>
    inline unccComplex<__Tp> operator/(const __oTp& rhs) {
        unccComplex<__Tp> z = *this;
        z /= rhs;
        return z;
    }

    template <typename __oTp>
    inline CUDAFUNCTION unccComplex<__Tp> operator*(const unccComplex<__oTp>& rhs) {
        unccComplex<__Tp> z = *this;
        z *= rhs;
        return z;
    }

    template <typename __oTp>
    inline CUDAFUNCTION unccComplex<__Tp> operator*(const __oTp& rhs) {
        unccComplex<__Tp> z = *this;
        z *= rhs;
        return z;
    }

    template <typename __oTp>
    inline CUDAFUNCTION unccComplex<__Tp> operator+(const unccComplex<__oTp>& rhs) {
        unccComplex<__Tp> z = *this;
        z += rhs;
        return z;
    }

    template <typename __oTp>
    inline unccComplex<__Tp> operator-(const unccComplex<__oTp>& rhs) {
        unccComplex<__Tp> z = *this;
        z -= rhs;
        return z;
    }

    template <typename __oTp>
    inline unccComplex<__Tp> operator=(const __oTp& other) {
        _M_real = other;
        _M_imag = 0;
        return *this;
    }

    template <typename __oTp>
    inline unccComplex<__Tp>& operator=(const unccComplex& other) {
        _M_real = other.real();
        _M_imag = other.imag();
        return *this;
    }

    inline CUDAFUNCTION unccComplex<__Tp>& operator+=(const unccComplex& rhs) {
        _M_real += rhs.real();
        _M_imag += rhs.imag();
        return *this;
    }

    template <typename __oTp>
    inline CUDAFUNCTION unccComplex<__Tp>& operator+=(const __oTp& rhs) {
        _M_real += rhs;
        return *this;
    }

    template <typename __oTp>
    inline unccComplex<__Tp>& operator-=(const __oTp& rhs) {
        _M_real -= rhs;
        return *this;
    }

    //template <typename __Tp>

    inline unccComplex<__Tp>& operator-=(const unccComplex<__Tp>& rhs) {
        _M_real -= rhs.real();
        _M_imag -= rhs.imag();
        return *this;
    }

    template <typename __oTp>
    inline CUDAFUNCTION unccComplex<__Tp>& operator*=(const __oTp& rhs) {
        _M_imag *= rhs;
        _M_real *= rhs;
        return *this;
    }

    inline CUDAFUNCTION unccComplex<__Tp>& operator*=(const unccComplex<__Tp>& rhs) {
        const float __r = _M_real * rhs._M_real - _M_imag * rhs._M_imag;
        _M_imag = _M_real * rhs.imag() + _M_imag * rhs.real();
        _M_real = __r;
        return *this;
    }

    inline unccComplex<__Tp>& operator/=(const unccComplex<__Tp>& rhs) {
        //std::cout << "this=" << *this << " rhs=" << rhs << std::endl;
        const __Tp __r = _M_real * rhs.real() + _M_imag * rhs.imag();
        const __Tp __n = norm(rhs);
        _M_imag = (_M_imag * rhs.real() - _M_real * rhs.imag()) / __n;
        _M_real = __r / __n;
        return *this;
    }

    template <typename __oTp>
    inline unccComplex<__Tp>& operator/=(const __oTp& rhs) {
        _M_real /= rhs;
        _M_imag /= rhs;
        return *this;
    }

    //friend _GLIBCXX_CONSTEXPR bool operator==(const mxComplexSingleClass& __x, const mxComplexSingleClass& __y);
    template <typename __oTp>
    friend std::ostream& operator<<(std::ostream& output, const unccComplex<__oTp>& c);
    template <typename __oTp>
    friend std::istream& operator>>(std::istream& input, const unccComplex<__oTp>& c);
};

template<typename __Tp>
inline unccComplex<__Tp> operator+(const unccComplex<__Tp>& __x, const unccComplex<__Tp>& __y) {
    unccComplex<__Tp> __r = __x;
    __r += __y;
    return __r;
}

template<typename __Tp>
inline unccComplex<__Tp> operator+(const unccComplex<__Tp>& __x, const __Tp& __y) {
    unccComplex<__Tp> __r = __x;
    __r += __y;
    return __r;
}

template<typename __Tp>
inline unccComplex<__Tp> operator+(const __Tp& __x, const unccComplex<__Tp>& __y) {
    unccComplex<__Tp> __r = __y;
    __r += __x;
    return __r;
}
//@}
//@{
///  Return new complex value @a x minus @a y.

template<typename __Tp>
inline unccComplex<__Tp> operator-(const unccComplex<__Tp>& __x, const unccComplex<__Tp>& __y) {
    unccComplex<__Tp> __r = __x;
    __r -= __y;
    return __r;
}

template<typename __Tp>
inline unccComplex<__Tp> operator-(const unccComplex<__Tp>& __x, const __Tp& __y) {
    unccComplex<__Tp> __r = __x;
    __r -= __y;
    return __r;
}

template<typename __Tp, typename __oTp>
inline unccComplex<__Tp> operator-(const __oTp& __x, const unccComplex<__Tp>& __y) {
    unccComplex<__Tp> __r(__x, -__y.imag);
    __r -= __y._M_real;
    return __r;
}

template<typename __Tp, typename __oTp>
inline unccComplex<__Tp> operator*(const unccComplex<__Tp>& __x, const unccComplex<__oTp>& __y) {
    unccComplex<__Tp> __r = __x;
    __r *= __y;
    return __r;
}

template<typename __Tp>
inline CUDAFUNCTION unccComplex<__Tp> operator*(const unccComplex<__Tp> & __x, const __Tp& __y) {
    unccComplex<__Tp> __r = __x;
    __r *= __y;
    return __r;
}

template<typename __Tp>
inline CUDAFUNCTION unccComplex<__Tp> operator*(const __Tp& __x, const unccComplex<__Tp>& __y) {
    unccComplex<__Tp> __r = __y;
    __r *= __x;
    return __r;
}

template<typename __Tp>
inline unccComplex<__Tp> operator/(const unccComplex<__Tp>& __x, const unccComplex<__Tp>& __y) {
    unccComplex<__Tp> __r = __x;
    __r /= __y;
    return __r;
}

template<typename __Tp>
inline unccComplex<__Tp> operator/(const unccComplex<__Tp>& __x, const __Tp& __y) {
    unccComplex<__Tp> __r = __x;
    __r /= __y;
    return __r;
}

template<typename __Tp>
inline unccComplex<__Tp> operator/(const __Tp& __x, const unccComplex<__Tp>& __y) {
    unccComplex<__Tp> __r = __x;
    __r /= __y;
    return __r;
}

// Templates for constant expressions with ==

template<typename __Tp>
inline _GLIBCXX_CONSTEXPR bool operator==(const unccComplex<__Tp>& __x, const unccComplex<__Tp>& __y) {
    return __x.real() == __y.real() && __x.imag() == __y.imag();
}

template<typename __Tp>
inline _GLIBCXX_CONSTEXPR bool operator==(const unccComplex<__Tp>& __x, const __Tp& __y) {
    return __x.real() == __y && __x.imag() == __Tp();
}

template<typename __Tp>
inline _GLIBCXX_CONSTEXPR bool operator==(const __Tp& __x, const unccComplex<__Tp>& __y) {
    return __x == __y.real() && __Tp() == __y.imag();
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

// Extend the std::abs namespace to be able to process type unccComplex<__Tp>
// We add std::abs(), std::polar(), and std::conj()

//template <typename __Tp>
//inline __Tp std::abs(unccComplex<__Tp>& __x) {
//    return __Tp();
//}
//
//template <typename __Tp>
//inline unccComplex<__Tp> std::polar(__Tp& __x, __Tp& __y) {
//    return unccComplex<__Tp>();
//}
//
//template <typename __Tp>
//inline unccComplex<__Tp> std::conj(unccComplex<__Tp>& __x) {
//    return unccComplex<__Tp>();
//}


// std::cout << "Enter a complex number (a+bi) : " << std::endl;
// std::cin >> x;

template <typename __Tp>
inline std::istream& operator>>(std::istream& input, unccComplex<__Tp>& __x) {
    char plus;
    char i;
    input >> __x._M_real >> plus >> __x.imag >> i;
    return input;
}
///  Insertion operator for complex values.

template <typename __Tp>
inline std::ostream& operator<<(std::ostream& output, const unccComplex<__Tp>& __x) {
    output << __x.real() << ((__x.imag() < 0) ? "" : "+") << __x.imag() << "i";
    return output;
}

#endif /* UNCC_COMPLEX_HPP */

