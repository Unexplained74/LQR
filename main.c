#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double** createMatrix(int rows, int cols) {
    // Allocate memory for the rows (pointers)
    double** matrix = malloc(rows * sizeof(double*));
    if (matrix == NULL) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }

    // For each row, allocate memory for the columns
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Out of memory\n");
            exit(1);
        }
    }

    // Now matrix is a 2D array of size rows x cols
    // Fill it with zeroes or any default value
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 0.0;
        }
    }

    return matrix;
}

void freeMatrix(double** matrix, int rows) {
    // Free each row
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    // Then free the array of rows
    free(matrix);
}

void readMatrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scanf_s("%lf", &matrix[i][j]);
        }
    }
}

void printMatrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        printf("[ ");
        for (int j = 0; j < cols; j++) {
            if (matrix[i][j] >= 0) {
                printf(" ");
            }
            printf("%lf ", matrix[i][j]);
        }
        printf("] \n");
    }
}

void transpose(double** matrix, int rows, int cols, double** result) {

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
}

double rand_range(double min, double max) {

    double random = ((double)rand()) / RAND_MAX;
    double range = (max - min) * random;
    double number = min + range;

    return number;
}

double** randomSqrMatrix(int dim, double min, double max) {
    // Allocate memory for the rows (pointers)
    double** matrix = malloc(dim * sizeof(double*));
    if (matrix == NULL) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }

    // For each row, allocate memory for the columns
    for (int i = 0; i < dim; i++) {
        matrix[i] = malloc(dim * sizeof(double));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Out of memory\n");
            exit(1);
        }
    }

    // Now matrix is a 2D array of size rows x cols
    // Fill it with zeroes or any default value
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            matrix[i][j] = rand_range(min, max);
        }
    }

    return matrix;
}

// Matrix multiplication with time complexity of O(n^3). (Can be more optimise)
void matrixMultiply(double** matrix1, int rows1, int cols1, double** matrix2, int rows2, int cols2, double** result) {
    if (cols1 != rows2) {
        fprintf(stderr, "\nWarning: Matrix sizes do not match! \n");
        exit(1);
    }
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

}

void matrixMultScaler(double** matrix, int rows, int cols, double scaler, double** result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrix[i][j] * scaler;
        }
    }
}

void sumMatrix(double** matrix1, double** matrix2, int rows, int cols, double** result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

void subtMatrix(double** matrix1, double** matrix2, int rows, int cols, double** result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
}

// Copy matrix 1 to matrix 2
void copyMatrix(double** matrix1, int rows, int cols, double** matrix2) {

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix2[i][j] = matrix1[i][j];
        }
    }
}

double** eye(int rows, int cols) {
    double** matrix = createMatrix(rows, cols);

    for (int i = 0; i < rows; i++) {
        matrix[i][i] = 1.0;
    }

    return matrix;
}

double dotProduct(double** vect1, int dim1, double** vect2, int dim2) {
    if (dim1 != dim2) {
        fprintf(stderr, "\nWarning: Dimensions not matching!\n");
        exit(1);
    }

    double result = 0.0;

    for (int i = 0; i < dim1; i++) {
        result += (vect1[i][0] * vect2[i][0]);
    }
    return result;
}

// Projection for Gram-Schmidt
void projection(double** vect1, int dim1, double** vect2, int dim2, double** result) {
    double gain = dotProduct(vect1, dim1, vect2, dim2) / dotProduct(vect2, dim2, vect2, dim2);

    for (int i = 0; i < dim1; i++) {
        result[i][0] = gain * vect2[i][0];
    }
}

double detMatrix(double** matrix, int rows, int cols) {

    if (rows != cols) {
        printf("\n  Warning: Determinant is not calculated, dimensions does not match! /n");
        exit(1);
    }

    // Create a temporary matrix
    double** tempMatrix = createMatrix(rows, 3 * cols);


    // Make an augmented matrix to ease the determinant process
    // It basically does: [tempMatrix] = [matrix|matrix|matrix]
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            tempMatrix[i][j] = matrix[i][j];
            tempMatrix[i][j + cols] = matrix[i][j];
            tempMatrix[i][j + (2 * cols)] = matrix[i][j];
        }
    }

    //printMatrix(tempMatrix, rows, 3 * cols);

    // We will return this
    double determinant = 0.0;

    for (int j = 0; j < cols; j++) {
        int c1 = j;         // Columns goes to right starts from the left
        int c2 = (3 * cols) - 1 - j;  // Columns goes to left stats from the right

        // Create two variable to hold intermediate transactions
        double multiplyPlus = 1.0;
        double multiplyMinus = 1.0;

        // This loop for goin down in the matrix
        for (int i = 0; i < rows; i++) {
            multiplyPlus *= tempMatrix[i][c1];      // Collect possitive terms by hoing right
            multiplyMinus *= tempMatrix[i][c2];     // Collect negative terms by going left

            c1++;   // This is for going right in the matrix
            c2--;   // This is for going left in the matrix
        }

        // Uncomment to see steps
        //printf("\nmultiplyPlus: %lf \n", multiplyPlus);
        //printf("multiplyMinus: %lf \n", multiplyMinus);

        determinant += multiplyPlus - multiplyMinus;    // Determinant is the sum of all terms (positive or negative)
    }

    // Delete the temporary matrix
    freeMatrix(tempMatrix, rows);


    return determinant;
}

// Function to invert a matrix using Gauss-Jordan elimination (My code)
/*
void invMatrix(double** matrix, int dim, double** inverse) {

    // Initialize "inverse" as an identity matrix
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            inverse[i][j] = 0.0;  // Initially, set all elements to zero
            if (i == j) {
                inverse[i][j] = 1.0;  // Diagonal elements are set to one
            }
        }
    }

    double temp;  // Temporary variable to hold various values

    // Forward pass: Making the matrix into an upper triangular one
    for (int i = 0; i < dim; i++) {
        temp = matrix[i][i];  // Diagonal element that current row will be divided by
        for (int j = 0; j < dim; j++) {
            matrix[i][j] /= temp;  // Normalize current row with its diagonal element
            inverse[i][j] /= temp;  // Do the same operations on the identity (will become the inverse)
        }

        // Subtract multiples of the normalized row from all lower rows
        for (int j = i + 1; j < dim; j++) {
            temp = matrix[j][i];  // Factor to multiply the i-th row before subtracting from j-th row
            for (int k = 0; k < dim; k++) {
                matrix[j][k] -= temp * matrix[i][k];  // Subtract from the main matrix
                inverse[j][k] -= temp * inverse[i][k];  // Do the same operations on the identity
            }
        }
    }

    // Backward pass: Make the matrix into diagonal one
    for (int i = dim - 1; i >= 0; i--) {
        temp = matrix[i][i];  // Diagonal element that current row will be divided by
        for (int j = dim - 1; j >= 0; j--) {
            matrix[i][j] /= temp;  // Normalize current row with its diagonal element
            inverse[i][j] /= temp;  // Do the same operations on the identity matrix
        }

        // Subtract multiples of the normalized row from all higher rows
        for (int j = i - 1; j >= 0; j--) {
            temp = matrix[j][i];  // Factor to multiply the i-th row before subtracting from j-th row
            for (int k = dim - 1; k >= 0; k--) {
                matrix[j][k] -= temp * matrix[i][k];  // Subtract from the main matrix
                inverse[j][k] -= temp * inverse[i][k];  // Do the same operations on the identity
            }
        }
    }
}
*/

// Function to invert a matrix using Gauss-Jordan elimination (My code improved by AI)
void invMatrix(double** matrix, int dim, double** inverse) {
    // Initialize "inverse" as an identity matrix
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            inverse[i][j] = 0.0;
            if (i == j) {
                inverse[i][j] = 1.0;
            }
        }
    }

    double temp;

    // Forward elimination
    for (int i = 0; i < dim; i++) {

        // Find the row with the largest value in the current column and swap it with the current row
        // This is called partial pivoting and helps to avoid numerical instability
        int maxRow = i;
        for (int j = i + 1; j < dim; j++) {
            if (fabs(matrix[j][i]) > fabs(matrix[maxRow][i])) {
                maxRow = j;
            }
        }
        double* tempRow = matrix[i];
        matrix[i] = matrix[maxRow];
        matrix[maxRow] = tempRow;
        tempRow = inverse[i];
        inverse[i] = inverse[maxRow];
        inverse[maxRow] = tempRow;

        temp = matrix[i][i];
        for (int j = 0; j < dim; j++) {
            // Scale the row so the diagonal element is 1
            matrix[i][j] /= temp;
            inverse[i][j] /= temp;
        }

        for (int j = i + 1; j < dim; j++) {
            // Eliminate other values in the column
            temp = matrix[j][i];
            for (int k = 0; k < dim; k++) {
                matrix[j][k] -= temp * matrix[i][k];
                inverse[j][k] -= temp * inverse[i][k];
            }
        }
    }

    // Backward substitution
    for (int i = dim - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            // Eliminate values above the diagonal
            temp = matrix[j][i];
            for (int k = dim - 1; k >= 0; k--) {
                matrix[j][k] -= temp * matrix[i][k];
                inverse[j][k] -= temp * inverse[i][k];
            }
        }
    }
}

double norm(double** vector, int dim) {
    double magnitude = 0.0;

    for (int i = 0; i < dim; i++) {
        magnitude += vector[i][0] * vector[i][0];
    }

    return sqrt(magnitude);

}

void normalize(double** vector, int dim) {

    double vectorNorm = norm(vector, dim);

    for (int i = 0; i < dim; i++) {
        vector[i][0] /= vectorNorm;
    }
}

// Only for square matrix
void qrdecomposition(double** A, double** Q, double** R, int dimA) {

    double** super_Q = eye(dimA, dimA);
    double** super_matrix = createMatrix(dimA, dimA);
    copyMatrix(A, dimA, dimA, super_matrix);

    for (int m = 0; m < (dimA - 1); m++) {
        int dim_temp = dimA - m;
        double** A_temp = createMatrix(dim_temp, dim_temp);

        for (int i = 0; i < dim_temp; i++) {
            for (int j = 0; j < dim_temp; j++) {
                A_temp[i][j] = super_matrix[m + i][m + j];
            }
        }

        double** Q_temp = createMatrix(dim_temp, dim_temp);
        double** a1 = createMatrix(dim_temp, 1);
        double** u = createMatrix(dim_temp, 1);
        double** v = createMatrix(dim_temp, 1);
        double** v_T = createMatrix(1, dim_temp);
        double** temp_matrix1 = createMatrix(dim_temp, dim_temp);
        double** temp_matrix2 = createMatrix(dim_temp, dim_temp);
        double** temp_matrix3 = eye(dimA, dimA);
        double** temp_matrix4 = eye(dimA, dimA);

        for (int i = 0; i < dim_temp; i++) {
            a1[i][0] = A_temp[i][0];
        }

        double alpha = norm(a1, dim_temp);

        for (int i = 0; i < dim_temp; i++) {
            if (i == 0) {
                u[i][0] = a1[i][0] - alpha;
            }
            else {
                u[i][0] = a1[i][0];
            }
            v[i][0] = u[i][0];
        }

        normalize(v, dim_temp);
        transpose(v, dim_temp, 1, v_T);
        matrixMultiply(v, dim_temp, 1, v_T, 1, dim_temp, temp_matrix1);
        matrixMultScaler(temp_matrix1, dim_temp, dim_temp, 2.0, temp_matrix2);
        temp_matrix1 = eye(dim_temp, dim_temp);
        subtMatrix(temp_matrix1, temp_matrix2, dim_temp, dim_temp, Q_temp);
        matrixMultiply(Q_temp, dim_temp, dim_temp, A_temp, dim_temp, dim_temp, temp_matrix1);

        for (int i = m; i < dimA; i++) {
            for (int j = m; j < dimA; j++) {
                temp_matrix3[i][j] = Q_temp[i - m][j - m];
            }
        }

        transpose(temp_matrix3, dimA, dimA, temp_matrix4);
        copyMatrix(super_Q, dimA, dimA, temp_matrix3);
        matrixMultiply(temp_matrix3, dimA, dimA, temp_matrix4, dimA, dimA, super_Q);

        // write it to super matrix
        for (int i = 1; i < dim_temp; i++) {
            for (int j = 1; j < dim_temp; j++) {
                super_matrix[m + i][m + j] = temp_matrix1[i][j];
            }
        }

        freeMatrix(A_temp, dim_temp);
        freeMatrix(Q_temp, dim_temp);
        freeMatrix(a1, dim_temp);
        freeMatrix(u, dim_temp);
        freeMatrix(v, dim_temp);
        freeMatrix(v_T, 1);
        freeMatrix(temp_matrix1, dim_temp);
        freeMatrix(temp_matrix2, dim_temp);
        freeMatrix(temp_matrix3, dimA);
        freeMatrix(temp_matrix4, dimA);
    }

    copyMatrix(super_Q, dimA, dimA, Q);
    transpose(Q, dimA, dimA, super_Q);
    matrixMultiply(super_Q, dimA, dimA, A, dimA, dimA, R);

    freeMatrix(super_Q, dimA);
    freeMatrix(super_matrix, dimA);
}

// Works best with the orthonormal eigenvectors
void qrAlgorithm(double** A, int dim, double** EigenValues, double** EigenVectors) {

    double** A_temp = createMatrix(dim, dim);
    double** Q = createMatrix(dim, dim);
    double** R = createMatrix(dim, dim);
    double** Q_temp = createMatrix(dim, dim);
    double** Q_accumulated = eye(dim, dim);
    copyMatrix(A, dim, dim, A_temp);

    int iteration = 5000;

    for (int i = 0; i < iteration; i++) {

        qrdecomposition(A_temp, Q, R, dim);
        matrixMultiply(R, dim, dim, Q, dim, dim, A_temp);

        matrixMultiply(Q_accumulated, dim, dim, Q, dim, dim, Q_temp);
        copyMatrix(Q_temp, dim, dim, Q_accumulated);

    }

    // cols
    for (int i = 0; i < dim; i++) {
        // rows
        for (int j = 0; j < dim; j++) {
            Q_accumulated[j][i] /= Q_accumulated[dim - 1][i];

        }
    }

    copyMatrix(A_temp, dim, dim, EigenValues);
    copyMatrix(Q_accumulated, dim, dim, EigenVectors);

    freeMatrix(A_temp, dim);
    freeMatrix(Q, dim);
    freeMatrix(R, dim);
    freeMatrix(Q_temp, dim);
    freeMatrix(Q_accumulated, dim);
}

void lqr(double** A, double** B, double** Q, double** R, int n, int m) {
    // Musts:
    // A -> n by n matrix
    // B -> n by m matrix
    // Q -> n by n matrix
    // R -> m by m matrix

    // Step1: Form Hamiltonian Matrix

    double** H = createMatrix(2 * n, 2 * n);
    double** H11 = createMatrix(n, n);
    double** H12 = createMatrix(n, n);
    double** H21 = createMatrix(n, n);
    double** H22 = createMatrix(n, n);
    double** B_temp1 = createMatrix(n, m);
    double** B_temp2 = createMatrix(n, m);
    double** B_transpose = createMatrix(m, n);
    double** R_temp = createMatrix(m, m);


    copyMatrix(A, n, n, H11);                                   // H11

    matrixMultScaler(B, n, m, -1.0, B_temp1);                   // H12
    transpose(B, n, m, B_transpose);
    invMatrix(R, m, R_temp);
    matrixMultiply(B_temp1, n, m, R_temp, m, m, B_temp2);
    matrixMultiply(B_temp2, n, m, B_transpose, m, n, H12);

    matrixMultScaler(Q, n, n, -1.0, H21);                       // H21

    transpose(A, n, n, H22);                                    // H22
    matrixMultScaler(H22, n, n, -1.0, H22);

    for (int i = 0; i < (2 * n); i++) {
        for (int j = 0; j < (2 * n); j++) {
            if ((0 <= i && i < n) && (0 <= j && j < n)) {
                H[i][j] = H11[i][j];
            }
            else if ((0 <= i && i < n) && (n <= j && j < (2 * n))) {
                H[i][j] = H12[i][j - n];
            }
            else if ((n <= i && i < (2 * n)) && (0 <= j && j < n)) {
                H[i][j] = H21[i - n][j];
            }
            else if ((n <= i && i < (2 * n)) && (n <= j && j < (2 * n))) {
                H[i][j] = H22[i - n][j - n];
            }
        }
    }

    // Step2: 

    double** EigVal = createMatrix(2 * n, 2 * n);
    double** EigVec = createMatrix(2 * n, 2 * n);

    qrAlgorithm(H, 2 * n, EigVal, EigVec);

    printf("\nH:\n");
    printMatrix(H, 2 * n, 2 * n);

    printf("\nEigen Values:\n");
    printMatrix(EigVal, 2 * n, 2 * n);

    printf("\nEigen Vectors:\n");
    printMatrix(EigVec, 2 * n, 2 * n);

    // Free all matrices
    freeMatrix(H, 2 * n);
    freeMatrix(H11, n);
    freeMatrix(H12, n);
    freeMatrix(H21, n);
    freeMatrix(H22, n);
    freeMatrix(B_temp1, n);
    freeMatrix(B_temp2, n);
    freeMatrix(B_transpose, m);
    freeMatrix(R_temp, m);
}


int main() {

    // Create A matrix
    int  rowsA = 4;
    int  colsA = 4;
    double** A = createMatrix(rowsA, colsA);
    // Create B matrix
    int  rowsB = 4;
    int  colsB = 1;
    double** B = createMatrix(rowsB, colsB);
    // Create Q matrix
    int  rowsQ = 4;
    int  colsQ = 4;
    double** Q = createMatrix(rowsQ, colsQ);
    // Create R matrix
    int  rowsR = 1;
    int  colsR = 1;
    double** R = createMatrix(rowsR, colsR);


    printf("A:\n");
    readMatrix(A, rowsA, colsA);

    printf("\nB:\n");
    readMatrix(B, rowsB, colsB);

    printf("\nQ:\n");
    readMatrix(Q, rowsQ, colsQ);

    printf("\nR:\n");
    readMatrix(R, rowsR, colsR);

    lqr(A, B, Q, R, rowsA, colsB);


    // Don't forget to free the matrix when you're done with it!
    freeMatrix(A, rowsA);
    freeMatrix(B, rowsB);
    freeMatrix(Q, rowsQ);
    freeMatrix(R, rowsR);


    return 0;
}
