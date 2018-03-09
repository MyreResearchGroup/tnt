% ====================================================================
% Least squares feasible solution using a preconditioned conjugate  
% gradient least-squares solver.
% ====================================================================
% Minimize norm2(b - A * x)
% ====================================================================
% Author: Erich Frahm, frahm@physics.umn.edu
%         Joseph Myre, myre@stthomas.edu
% ====================================================================

function [ x, score, residual, AA, epsilon] = tnt(A, b)

status = 3; % unknown failure
iteration = -1;

% Get the input matrix size.
[m, n] = size(A);

% Check the input vector size.
[mb, nb] = size(b);
if ((mb ~= m) || (nb ~= 1))
    status = 2; % failure: vector is wrong size
    return;
end

% ===============================================================
% AA is a symmetric and positive definite (probably) n x n matrix.
% If A did not have full rank, then AA is positive semi-definite.
% Also, if A is very ill-conditioned, then rounding errors can make 
% AA appear to be indefinite. Modify AA a little to make it more
% positive definite.
% ===============================================================
% Compute normal equations
AA = A'*A;
epsilon = 10 * eps(1) * norm(AA,1);
AA = AA + (epsilon * eye(n));

% =============================================================
% Cholesky decomposition.
% =============================================================
[R,p] = chol(AA);
while (p > 0)
    % It may be necessary to add to the diagonal of B'B to avoid 
    % taking the sqare root of a negative number when there are 
    % rounding errors on a nearly singular matrix. That's still 
	% OK because we just use the Cholesky factor as a 
	% preconditioner.
    epsilon = epsilon * 10;
    AA = AA + (epsilon * eye(n));

	[R, p] = chol(AA);
end

% ------------------------------------------------------------
% Use PCGNR to find the unconstrained optimum in 
% the "free" variables.
% ------------------------------------------------------------
[x, k] = pcgnr(A,b,R);

% ------------------------------------------------------------
% Compute the full (unreduced) residual.
% ------------------------------------------------------------
residual = b - (A * x);

% ------------------------------------------------------------
% Compute the norm of the residual.
% ------------------------------------------------------------
score = sqrt(dot(residual,residual));

return;
end

% ====================================================================
% Iterative Methods for Sparse Linear Systems, Yousef Saad
% Algorithm 9.7 Left-Preconditioned CGNR
% http://www.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf
% ====================================================================
% Author: Erich Frahm, frahm@physics.umn.edu
% ====================================================================

function [ x, k ] = pcgnr ( A, b, R )
    [ m, n ] = size(A);
    x = zeros(n,1);
    r = b;
    r_hat = A' * r; % matrix_x_vector, O(mn)
    y = R' \ r_hat; % back_substitution, O(n^2)
    z = R \ y; % back_substitution, O(n^2)
    p = z;
    gamma = dot(z,r_hat);
    prev_rr = -1;
    for k = 1:n
        w = A * p; % matrix_x_vector, O(mn)
        ww = dot(w,w);
        if (ww == 0)
            return;
        end
        alpha = gamma/ww;
        x_prev = x;
        x = x + (alpha*p);
        r = b - (A * x); % matrix_x_vector, O(mn)
        r_hat = A' * r; % matrix_x_vector, O(mn)
        
        % ---------------------------------------------
        % Enforce continuous improvement in the score.
        % ---------------------------------------------
        rr = dot(r_hat,r_hat);
        if ((prev_rr >= 0) && (prev_rr <= rr))
            x = x_prev;
            return;
        end
        prev_rr = rr;
        % ---------------------------------------------
        
        y = R' \ r_hat; % back_substitution, O(n^2)
        z = R \ y; % back_substitution, O(n^2)
        gamma_new = dot(z,r_hat);
        beta = gamma_new / gamma;
        p = z + (beta * p);
        gamma = gamma_new;
        if (gamma == 0)
            return;
        end
    end
end
