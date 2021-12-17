#include "math.h"


//----------------------------------------
// Fitting Tools
// ---------------------------------------

// based on
// https://stackoverflow.com/questions/11449617/how-to-fit-the-2d-scatter-data-with-a-line-with-c
std::vector<double> FitLine(const std::vector<cv::Point>& p, bool invert=false) {    
    int nPoints = p.size();
    if( nPoints < 2 ) {
        // Fail: infinitely many lines passing through this single point
        return std::vector<double>();
    }

    // invert: x = f(y)
    if (invert){
        double sumX=0, sumY=0, sumXY=0, sumY2=0;
        for(int i=0; i<nPoints; i++){
            sumX += p[i].x;
            sumY += p[i].y;
            sumXY += p[i].x * p[i].y;
            sumY2 += p[i].x * p[i].x;        
        }        
        double xMean = sumX / nPoints;
        double yMean = sumY / nPoints;
        double denominator = sumY2 - sumY * yMean;
        if( std::fabs(denominator) < 1e-7 ) {
            // Fail:  vertical line
            return std::vector<double>();
        }
        double slope = (sumXY - sumY * xMean) / denominator;
        double x0 = xMean - slope * yMean;
        // line x0 + slope * yy
        return std::vector<double>(x0, slope);
    
    } else {
        
        // usual case: y = f(x)
        double sumX=0, sumY=0, sumXY=0, sumX2=0;
        for(int i=0; i<nPoints; i++) {
            sumX += p[i].x;
            sumY += p[i].y;
            sumXY += p[i].x * p[i].y;
            sumX2 += p[i].x * p[i].x;        
        }

        // invert: x = f(y)
        double xMean = sumX / nPoints;
        double yMean = sumY / nPoints;
        double denominator = sumX2 - sumX * xMean;
        if( std::fabs(denominator) < 1e-7 ) {
        // Fail:  vertical line
        return std::vector<double>();
        }
        double slope = (sumXY - sumX * yMean) / denominator;
        double y0 = yMean - slope * xMean;
        // line y0 + slope *xx
        return std::vector<double>(y0, slope);        
    }
}

// polynomial regression: based on
//https://rosettacode.org/wiki/Polynomial_regression#C.2B.2B
std::vector<double> FitParabola(const std::vector<cv::Point>& p, bool invert=false) {
    int n = p.size();
    double xm=0, ym=0, x2m=0, x3m=0, x4m=0; // means of moments
    double xym=0, x2ym=0; // menans of cross moments

    if (invert){ // invert: swap x for y
        for(int i=0; i<n; i++){
            xm += p[i].y;
            ym += p[i].x;

            double x2 = p[i].y * p[i].y;
            x2m += x2;
            x3m += x2*p[i].y;
            x4m += x2*x2;

            xym  += p[i].x*p[i].y;
            x2ym += x2*p[i].x;
        }
    } else{ // usual case: y = f(x)
        for(int i=0; i<n; i++){
            xm += p[i].x;
            ym += p[i].y;

            double x2 = p[i].x * p[i].x;
            x2m += x2;
            x3m += x2*p[i].x;
            x4m += x2*x2;

            xym  += p[i].x*p[i].y;
            x2ym += x2*p[i].y;
        }
    }
    // divide by number of points (x stands for abscissa in this case y = f(x))
    xm/=n;
    ym/=n;
    x2m/=n;
    x3m/=n;
    x4m/=n;
    xym/=n;
    x2ym/=n;
 
    double sxx = x2m - xm * xm;
    double sxy = xym - xm * ym;
    double sxx2 = x3m - xm * x2m;
    double sx2x2 = x4m - x2m * x2m;
    double sx2y = x2ym - x2m * ym;
 
    double b = (sxy * sx2x2 - sx2y * sxx2) / (sxx * sx2x2 - sxx2 * sxx2);
    double c = (sx2y * sxx - sxy * sxx2) / (sxx * sx2x2 - sxx2 * sxx2);
    double a = ym - b * xm - c * x2m;
 
    // parabola: a + b * xx + c * xx*xx;
    return std::vector<double>{a, b, c};
}

// https: // stackoverflow.com / questions / 5477359 / chi - square - numpy - polyfit - numpy
double chi_squared(const std::vector<double> & poly_cfs, const std::vector<cv::Point>& p, bool invert=false){
    double ch_sqr=0.0;
    int order_poly = poly_cfs.size() - 1;
    int n_pts = p.size();

    if(invert){ // x = f(y)
        for(int i = 0; i<n_pts; ++i){
            double poly_eval{poly_cfs[0]};
            for (int k = 1; k < (order_poly + 1); ++k) 
                poly_eval += poly_cfs[k]*std::pow(p[i].y, k);
            ch_sqr += std::pow(poly_eval - p[i].x, 2);
        } // add over the points
    } else {    // y = f(x)
        for(int i = 0; i<n_pts; ++i){
            double poly_eval{poly_cfs[0]};
            for (int k = 1; k < (order_poly + 1); ++k)
                poly_eval += poly_cfs[k]*std::pow(p[i].x, k);
            ch_sqr += std::pow(poly_eval - p[i].y, 2);
        } // add over the points
    }
    return ch_sqr;
}

//----------------------------------------
// Evaluate Fit in cv::Point format
// ---------------------------------------

// evaluation for float and integer (plotting tool)
template <typename T>
std::vector<T> EvalFit(const std::vector<double> & poly_cfs, const std::vector<cv::Point>& p, bool invert){
    std::vector<T> eval;
    int order_poly = poly_cfs.size() - 1;
    int n_pts = p.size();

    if(invert){ // x = f(y)
        for(int i = 0; i<n_pts; ++i){
            double poly_eval{poly_cfs[0]};
            for (int k = 1; k < (order_poly + 1); ++k) 
                poly_eval += poly_cfs[k]*std::pow(p[i].y, k);     
            eval.emplace_back(static_cast<T>(poly_eval));
        } // add over the points
    } else {    // y = f(x)
        for(int i = 0; i<n_pts; ++i){
            double poly_eval{poly_cfs[0]};
            for (int k = 1; k < (order_poly + 1); ++k)
                poly_eval += poly_cfs[k]*std::pow(p[i].x, k);
            eval.emplace_back(static_cast<T>(poly_eval));
        } // add over the points
    }
    return eval; 
}



// explicit instantiation (declaration of explicit instance) for cv::Point ==> [-fpermissive]
//template std::vector<cv::Point> EvalFit<cv::Point>(const std::vector<double> & poly_cfs, 
//                                                   const std::vector<cv::Point>& p, bool invert);

// explicit specialization (definition of explicit instance) for cv::Point
template <> std::vector<cv::Point> EvalFit<cv::Point>(const std::vector<double> & poly_cfs, 
                                                      const std::vector<cv::Point>& p, bool invert){
    std::vector<cv::Point> eval;
    int order_poly = poly_cfs.size() - 1;
    int n_pts = p.size();

    if(invert){ // x = f(y)
        for(int i = 0; i<n_pts; ++i){
            double poly_eval{poly_cfs[0]};
            for (int k = 1; k < (order_poly + 1); ++k) 
                poly_eval += poly_cfs[k]*std::pow(p[i].y, k);     
            eval.emplace_back(cv::Point(static_cast<int>(poly_eval), p[i].y));
        } // add over the points
    } else {    // y = f(x)
        for(int i = 0; i<n_pts; ++i){
            double poly_eval{poly_cfs[0]};
            for (int k = 1; k < (order_poly + 1); ++k)
                poly_eval += poly_cfs[k]*std::pow(p[i].x, k);
            eval.emplace_back(cv::Point(p[i].x, static_cast<int>(poly_eval)));
        } // add over the points
    }
    return eval; 
}
//----------------------------------------


//----------------------------------------
// Implementation of BufferStats class
// ---------------------------------------

// compute mean and std deviation for a buffer of N std::vectors<T>
template<typename T>
BufferStats<T>::BufferStats(int n):n_buffer(n), idx_buffer(-1){
    n_buffer = n;
}

template<typename T>
void BufferStats<T>::add(const std::vector<T> & x){ // add measurement x to the stats
    if(idx_buffer == -1){
        n_pts = x.size();
        k_ = x;
        dev_ = std::vector<T>(n_pts);
        dev2_ = std::vector<T>(n_pts);
    } // first measurement

    // process only a valid input
    assert(static_cast<size_t>(n_pts) == x.size());

    ++idx_buffer;
    for(int i =0; i<n_pts; ++i){
        dev_[i]  += x[i] - k_[i];
        dev2_[i] += (x[i] - k_[i])*(x[i] - k_[i]);
    }
}

template<typename T>
void BufferStats<T>::remove(const std::vector<T> & x){  // remove measurement x from the stats
    --idx_buffer;
    for(int i =0; i<n_pts; ++i){
        dev_[i]  -= x[i] - k_[i];
        dev2_[i] -= (x[i] - k_[i])*(x[i] - k_[i]);
    }
}

template<typename T>
std::vector<T> BufferStats<T>::mean(){
    std::vector<T> mn;
    if(idx_buffer > -1){
        mn = std::vector<T>(n_pts);
        for(int i =0; i<n_pts; ++i){
            mn[i]  = k_[i] + dev_[i]/(idx_buffer+1);            
        } 
    }
    return mn; // empty if no points!
}

template<typename T>
std::vector<T> BufferStats<T>::stddev(){
    std::vector<T> sdv; 
    if(idx_buffer > 0){
        sdv = std::vector<T>(n_pts);
        for(int i =0; i<n_pts; ++i){
            sdv[i] = std::sqrt( (dev2_[i] - (dev_[i] * dev_[i]) / (idx_buffer+1)) / (idx_buffer) );
        } 
    }
    return sdv; // empty if less than 2 points!
}

template<typename T>
bool BufferStats<T>::is_outlier(const std::vector<T> & x){
    bool is_outlier{false}; 
    if(idx_buffer > 0){
        std::vector<T> mn = this->mean(); 
        std::vector<T> sdv = this->stddev(); 
        for(int i =0; i<n_pts; ++i){
            is_outlier = (is_outlier || std::fabs(x[i] - mn[i]) > sdv[i]);
        } 
    }
    return is_outlier; 
}


// implementation notes: 
// https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file

// explicit instantiation for needed types
template class BufferStats<double>;
template class BufferStats<int>;
// ---------------------------------------
