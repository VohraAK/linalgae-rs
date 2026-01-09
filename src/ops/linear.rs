use std::ops;
use crate::core::matrix::Matrix;


//----------Linear Algebra Operations----------//

// inner product / matmul
impl ops::Mul<&Matrix> for &Matrix
{
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output 
    {
        let lhs_cols = self.cols();
        let rhs_rows = rhs.rows();
        
        // check dims
        if lhs_cols != rhs_rows
        {
            panic!("Matrix::Mul: lhs_cols ({}) != rhs_rows ({})", lhs_cols, rhs_rows);
        }
        
        let lhs_rows = self.rows();
        let rhs_cols = rhs.cols();

        let lhs_slice = self.as_slice();
        let rhs_slice = rhs.as_slice();

        let mut result = vec![0 as f64; lhs_rows * rhs_cols];

        // TODO: improve naive O(N^3) implementation
        for i in 0..lhs_rows
        {
            for j in 0..rhs_cols
            {
                let mut dot_product = 0.0;

                for k in 0..lhs_cols
                {
                    // arr_1[i][k] * arr_2[k][j]
                    let lhs_idx = i*lhs_cols + k;
                    let rhs_idx = k*rhs_cols + j;

                    dot_product += lhs_slice[lhs_idx] * rhs_slice[rhs_idx];
                }

                // assign dot product to result
                let res_idx = i*rhs_cols + j;
                result[res_idx] = dot_product;
            }
        }

        Matrix::new(lhs_rows, rhs_cols, result).expect("Matrix::Mul: Result has incorrect dimensions!")
    }
}