#ifndef TNT_H
#define TNT_H

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

using namespace Eigen;


/* Main call */ 
VectorXd tnt(Ref<MatrixXd> A, Ref<VectorXd> b);

/** Helper **/
VectorXd pcgnr(Ref<MatrixXd> A, Ref<VectorXd> b, Ref<MatrixXd> R); 


#endif
/* vim: set sw=4 sts=4 et foldmethod=syntax syntax=c : */
