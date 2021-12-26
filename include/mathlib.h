#ifndef MATH_H
#define MATH_H

#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <queue>
#include <utility>
#include <cassert>
#include "opencv2/core.hpp"

std::vector<double> FitLine(const std::vector<cv::Point>& p, bool invert);
std::vector<double> FitParabola(const std::vector<cv::Point>& p, bool invert);
double chi_squared(const std::vector<double> & poly_cfs, const std::vector<cv::Point>& p, bool invert);

double r_curve(std::vector<double>polycoef, float y);

// implementation notes: 
// https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file

// allow returning e.g. double or int, as needed
template <typename T>
std::vector<T> EvalFit(const std::vector<double> & poly_cfs, const std::vector<cv::Point>& p, bool invert);


template<typename T>
class BufferStats{
// compute statistics of a vector of n values over a buffer of n_buffer realizations
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    private:
        int n_buffer, idx_buffer;
        int n_pts;
        std::vector<T> k_, dev_, dev2_;  // first value, deviations and squared dev

    public:
        BufferStats() = delete;
        BufferStats(int n_buffer); // buffer size parameter is mandatory

        void add(const std::vector<T> & x);
        void remove(const std::vector<T> & x);
        std::vector<T> mean();
        std::vector<T> stddev();
        std::vector<T> stddev(float sigma);
        bool is_outlier(const std::vector<T> & x, float sigma_factor = 1.0);
        bool has_NaN(const std::vector<T> & x);
};

#endif