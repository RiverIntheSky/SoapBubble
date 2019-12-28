# pragma once

# include "cuda_runtime_api.h"
# include <vector>
# include <algorithm>

__inline__ __host__ __device__ double2 operator*(double2 a, double b) {
    return make_double2(a.x * b, a.y * b);
}

__inline__ __host__ __device__ double2 operator*(double b, double2 a) {
    return make_double2(a.x * b, a.y * b);
}

__inline__ __host__ __device__ double3 operator*(double3 a, double b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}

__inline__ __host__ __device__ double3 operator*(double b, double3 a) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}

__inline__ __host__ __device__ double2 operator+(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}

__inline__ __host__ __device__ double3 operator+(double3 a, double3 b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__inline__ __host__ __device__ double2 operator-(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}

__inline__ __host__ __device__ double3 operator-(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__inline__ __host__ __device__ double2 operator/(double2 a, double b) {
    return make_double2(a.x / b, a.y / b);
}

__inline__ __host__ __device__ double3 operator/(double3 a, double b) {
    return make_double3(a.x / b, a.y / b, a.z / b);
}

__inline__ __host__ __device__ void operator*=(double2 &a, double b) {
    a.x *= b;
    a.y *= b;
}

__inline__ __host__ __device__ double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__inline__ __host__ __device__ double length(double3 v) {
    return sqrt(dot(v, v));
}

template <typename T>
inline T dot(std::vector<T>& a, std::vector<T>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__inline__ __host__ __device__ double3 cross(double3 a, double3 b) {
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

template <typename T>
inline std::vector<T> cross(std::vector<T>& a, std::vector<T>& b) {
    std::vector<T> c = {a[1] * b[2] - a[2] * b[1],
			a[2] * b[0] - a[0] * b[2],
			a[0] * b[1] - a[1] * b[0]};
    return c;
}

__inline__ __host__ __device__ double3 normalize(double3 v) {
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}

template <typename T>
inline std::vector<T> normalized(std::vector<T> v) {
    T invLen = 1.0 / sqrt(dot(v, v));
    std::vector<T> result;
    result.reserve(3);
    result[0] = v[0] * invLen;
    result[1] = v[1] * invLen;
    result[2] = v[2] * invLen;
    return result;
}

template <typename T>
inline void normalize(std::vector<T>& v) {
    T invLen = 1.0 / sqrt(dot(v, v));
    v[0] *= invLen;
    v[1] *= invLen;
    v[2] *= invLen;
}

__inline__ __host__ __device__ double safe_acos(double x) {
    return acos(max(-1.0, min(1.0, x)));
}

template<class T>
__inline__ __host__ __device__ const T& clamp( const T& v, const T& lo, const T& hi ) {
    assert(!(hi < lo));
    return (v < lo) ? lo : (hi < v) ? hi : v;
}
    
__inline__ __host__ __device__ float safe_acos(float x) {
    return acosf(clamp(x, -1.f, 1.f));
}

