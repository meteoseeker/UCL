function [V,D,L] = spectraleigs2(S,k,lapNorm)
%SPECTRALEIGS Spectrum of eigenvectors and eigenvalues.
%   V = SPECTRALEIGS(S, K) accepts a similarity matrix S. S must be a
%   square and symmetric matrix of size N-by-N, where N is the number of
%   observations. It returns the eigenvectors V corresponding to the
%   K-smallest eigenvalues of the graph Laplacian matrix. V is an N-by-K
%   matrix.
%
%   [V, D] = SPECTRALEIGS(...) additionally returns the K-smallest
%   eigenvalues D of the graph Laplacian Matrix as a K-by-1 vector
%
%   [V, D, L] = SPECTRALEIGS(...) additionally returns the graph Laplacian
%   Matrix L. L is an N-by-N matrix and depends on the choice of
%   LaplacianNormalization
%
%   'LaplacianNormalization' -  Method to normalize the
%                               Laplacian matrix L which is used to compute
%                               eigenvectors. Choices are:
%          'randomwalk'      -  Normalize L by inv(D_g)*L as in Shi et al.
%                               [1].
%                               D_g is the degree matrix. (default)
%          'symmetric'       -  Normalize L by D_g^(-1/2)*L*D_g^(-1/2) as 
%                               in Ng et al. [2].
%          'none'            -  Use L without normalization.
%
%
%   Example:
%      % Find eigenvectors of data X, using the default distance metric 
%      % 'euclidean'.
%      X = [rand(20,2)+2; rand(20,2)];
%      S = internal.stats.similarity(X);
%      V = internal.stats.spectraleigs(S,2);
%
%   See also SPECTRALCLUSTER, EIGS.
%
%   References
%   [1] Shi, J., and J. Malik. "Normalized cuts and image segmentation."
%   IEEE Transactions on Pattern Analysis and Machine Intelligence. Vol.
%   22, 2000, pp. 888-905.
%   [2] Ng, A.Y., M. Jordan, and Y. Weiss. "On spectral clustering: Analysis
%   and an algorithm." In Proceedings of the Advances in Neural Information
%   Processing Systems 14. MIT Press, 2001, pp. 849-856.
%   [3] Von Luxburg, U. "A Tutorial on Spectral Clustering." Statistics and
%   Computing Journal. Vol.17, Number 4, 2007, pp.395-416. 

%   Copyright 2019 The MathWorks, Inc.

if nargin < 3
    lapNorm = 'randomwalk';
else
    lapNorm = convertStringsToChars(lapNorm);
end

funcName = mfilename;
lapNorm = validatestring(lapNorm,{'randomwalk','symmetric','none'}...
    ,funcName,'LaplacianNormalization');

% Validate Similarity Matrix
validateattributes(S,{'single','double'},{'real','square','nonnegative',...
    'nonempty','nonnan','finite'},funcName,'Similarity Matrix');
if ~issymmetric(S)
    error(message('stats:spectraleigs:InvalidSimMat'));
end

% Validate k
validateattributes(k,{'single','double'},{'scalar','integer','nonempty',...
    'nonsparse','positive'},funcName,'k');
if k > size(S,1)
    error(message('stats:spectraleigs:TooManyEigs'));
end

% Start vector for eigs. Pass a start vector so that the results from eigs
% are reproducible and can be controlled by the MATLAB random seed
sv = randn(size(S,1),1);

% Compute Laplacian, Eigenvectors and Eigenvalues based on
% LaplacianNormalization
switch lapNorm
    case 'none'
        [V,D,L] = unnormalized(S,k,sv);
    case 'randomwalk'
        [V,D,L] = shimalik(S,k,sv);
    case 'symmetric'
        [V,D,L] = ngjordanweiss(S,k,sv);
end
end

% Get the Degree and unnormalized Laplacian matrices
function [L,D] = getDegreeAndLaplacian(L)
% The Similarity Matrix here: L = S 
% Set the diagonal elements to 0. The diagonal elements do not affect the
% unnormalized Laplacian matrix but do affect the degree matrix
L(1:size(L,1)+1:end) = 0;

% Construct the degree matrix. No need to diagonalize the matrix, keep it
% as a vector. D is defined as di = sum_j(Wij)
D = sum(L,2);

% Unnormalized Laplacian
L = - L;

if issparse(L)
    % Assignment of D to diagonal entries of a sparse matrix can be slow.
    % Instead, just add diag(D) to L
    L = L + diag(D);
else
    L(1:size(L,1)+1:end) = D;
end
end

% Unnormalized algorithm
function [V,D,L] = unnormalized(S,k,sv)
% 1. Get the Degree and unnormalized-Laplacian matrices
L = getDegreeAndLaplacian(S);

% 2. Compute first k eigenvectors of L
[V,D] = eigs(L,k,1e-10,'StartVector',sv);

% eigs returns a diagonal matrix D. Return just the eigenvalues as a vector
D = diag(D);
end

% Shi-Malik algorithm
function [V,D,L] = shimalik(S,k,sv)
% 1. Get the Degree and unnormalized Laplacian matrices
[L,Deg] = getDegreeAndLaplacian(S);

% The Shi-Malik Laplacian is defined as inv(Deg)*L
% Since Deg is diagonal, its inverse is merely the reciprocal of the
% diagonal elements. When diagonal elements are 0 (or near 0) set their
% values to 0 (pseudo-inverse).
Deg = invertDegreeMatrix(Deg);

% Normalize the Laplacian. We could have solved the generalized eigenvalue
% problem (L*v = lambda*Deg*v) instead of inverting Deg. However, this
% does not produce good results when Deg contains very small diagnonal
% elements
L = Deg.*L;

% Remove sub-normal numbers if they exist
L = setSubnormalNumsToZero(L);

% 2. Compute first k eigenvectors of L
[V,D] = eigs(L,k,1e-10,'StartVector',sv);

% eigs returns a diagonal matrix D. Return just the eigenvalues as a vector
D = diag(D);
end

% Ng-Jordan-Weiss algorithm
function [V,D,L] = ngjordanweiss(S,k,sv)
% 1. Get the Degree and unnormalized Laplacian matrices
[L,Deg] = getDegreeAndLaplacian(S);

% 2. The original algorithm normalizes the unnormalized Laplacian by: 
% L = D^-1/2*L*D^-1/2;
% However, D^-1/2 could lead to warnings if there are isolated vertices in
% the graph. Instead, set the Deg of isolated vertices to 0 post inversion

% Update Deg to Deg^(-1/2) using the pseudo-inverse concept
Deg = invertDegreeMatrix(Deg);
Deg = sqrt(Deg);

% Normalize the Laplacian by multiplying Deg on each side of L
L = Deg.*L.*Deg';

% Remove sub-normal numbers if they exist
L = setSubnormalNumsToZero(L);

% 2. Compute first k eigenvectors of L
[V,D] = eigs(L,k,1e-10,'StartVector',sv);

% 4. Normalize V so that rows sum to 1
vnorm = vecnorm(V,2,2); % Use vecnorm for improved numerical precision
nonZeroVNorm = vnorm > 0;
vnorm(~nonZeroVNorm) = 0;
if any(nonZeroVNorm)
    V(nonZeroVNorm,:) = V(nonZeroVNorm,:)./vnorm(nonZeroVNorm);
end

% eigs returns a diagonal matrix D. Return just the eigenvalues as a vector
D = diag(D);
end

% Subfunction for degree matrix inversion (pseudo-inverse)
function d = invertDegreeMatrix(d)
N = numel(d);
isTooSmall = d <= N*eps(max(d));
d = 1./d;
d(isTooSmall) = 0;
end

% Subfunction to remove subnormal numbers from Laplacian matrix
function L = setSubnormalNumsToZero(L)
N = numel(L);
isTooSmall = abs(L) <= N*eps(max(max(L))) & (abs(L)>0);
L(isTooSmall) = 0;
end