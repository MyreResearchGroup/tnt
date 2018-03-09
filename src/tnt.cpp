#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <random>
#include <sys/time.h>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "tnt.h"

using namespace Eigen;

VectorXd tnt(Ref<MatrixXd> A, Ref<VectorXd> b){

    /** Compute normal equations for preconditioner
     *
     * AA is a symmetric and positive definite (probably) n x n matrix.
     * If A did not have full rank, then AA is positive semi-definite.
     * Also, if A is very ill-conditioned, then rounding errors can make 
     * AA appear to be indefinite. Modify AA a little to make it more
     * positive definite.
     **/

	MatrixXd AA = A.transpose()*A;

    /** Used to ensure positive definite-ness **/
    double epsilon = 10 * std::numeric_limits<double>::epsilon() * AA.norm();


    /** 
     * =============================================================
     * Cholesky decomposition - for preconditioning.
     * ============================================================= 
     **/

    LLT<MatrixXd> lltofAA(AA);
    
    /** 
     * It may be necessary to add to the diagonal of A^{T}A to avoid 
     * taking the sqare root of a negative number when there are 
     * rounding errors on a nearly singular matrix. That's still 
     * OK because we just use the Cholesky factor as a preconditioner.
     **/
    while(lltofAA.info() == Eigen::NumericalIssue){
        epsilon *= 10;
        AA.diagonal().noalias() = AA.diagonal()+(epsilon*VectorXd::Ones(AA.cols()));

        // Update the Cholesky factorization
        lltofAA.compute(AA);
    }

    

    /** 
     * =============================================================
     * Preconditioned Conjugate Gradient Normal Residual
     * ============================================================= 
     **/

    MatrixXd R = lltofAA.matrixU();
    
    VectorXd x = pcgnr(A, b, R);

	return x;
}

/***
 *
 * PCGNR
 *
 * Based on Iterative Methods for Sparse Linear Systems, Yousef Saad
 *  Algorithm 9.7 Left-Preconditioned CGNR
 *   http://www.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf
 *  ====================================================================
 *  Author: Joseph Myre, myre@stthomas.edu
 *  ====================================================================
 *
 ***/
VectorXd pcgnr(Ref<MatrixXd> A, Ref<VectorXd> b, Ref<MatrixXd> R){

    int m = A.rows();
    int n = A.cols();

    int itr_limit = 1000;

    VectorXd x = VectorXd::Zero(n);
    VectorXd x_prev;
    VectorXd r(b);
    VectorXd r_hat = A.transpose() * r;

    VectorXd y(r_hat);
    R.transpose().triangularView<Eigen::Lower>().solveInPlace(y); 
    VectorXd z(y);
    R.triangularView<Eigen::Upper>().solve(z);
    
    VectorXd p(z); 
    VectorXd w;

    double gamma = z.dot(r_hat);
    double gamma_new = gamma;
    double ww = 0.;
    double alpha = 0.;
    double beta = 0.;

    double rr = 0.;
    double prev_rr = -1.;

    int ww_lim = 0.00000001*std::numeric_limits<double>::epsilon();

    for(int i = 0; i < itr_limit; i++){
        
        w = A*p;
        ww = w.dot(w);

        if((ww > -1*ww_lim) && (ww < ww_lim)){
            break;
        } 

        alpha = gamma/ww;
        
        x_prev = x; 

        x = x + (alpha*p);

        r = b - (A*x);
        r_hat = A.transpose()*r;

        /**
         * Enforce improvement in the score
         **/
        rr = r_hat.dot(r_hat);
        if((prev_rr >= 0) && (prev_rr <= rr)){
            x = x_prev;
            break;
        }
        prev_rr = rr;

        y = r_hat;
        R.transpose().triangularView<Eigen::Lower>().solveInPlace(y);
        z = y;
        R.triangularView<Eigen::Upper>().solveInPlace(z);
        gamma_new = z.dot(r_hat);
        beta = gamma_new / gamma;
        p = z + (beta * p);
        gamma = gamma_new;
        if(gamma == 0){
            break;
        }
    }

    return x;

}

/* vim: set sw=4 sts=4 et foldmethod=syntax syntax=c : */
